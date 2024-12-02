import math
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from pandas import Series
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc

import auto_encoder as ae
import data_loader
from anomaly import model_name as used_dataset
from anomaly import model_path, device
from transformer import TransformerClassifier, get_optimal_threshold

app = Flask(__name__)

X_train_scaled, X_test_scaled, Y_train, Y_test = data_loader.load_cic_unsw_data(binary=True)
train_loader, test_loader = data_loader.create_dataloaders(X_train_scaled, X_test_scaled, use_gpu=torch.cuda.is_available())
class_train_loader, class_test_loader = data_loader.create_classification_dataloaders(X_train_scaled, X_test_scaled, Y_train, Y_test)
data_generator = data_loader.get_ctgan_generator(X_train_scaled, Y_train, model_name=used_dataset)
label_column_name = data_loader.cic_unsw_label_name

MODEL_PATHS = {
    'Isolation Forest': f'{model_path}/iso_forest.pkl',
    'Local Outlier Factor': f'{model_path}/lof.pkl',
    'One-Class SVM': f'{model_path}/ocsvm.pkl',
    'K-Means': f'{model_path}/kmeans.pkl',
    'Autoencoder': f'{model_path}/autoencoder.pth',
    'Transformer': f'{model_path}/transformer.pth',
}

loaded_models = {}

PLOT_DIR = 'static/plots/'
os.makedirs(PLOT_DIR, exist_ok=True)


def load_model(model_name):
    """
    Loads a model from the model paths, caches it in memory after the first time it's loaded.
    """
    if model_name in loaded_models:
        return loaded_models[model_name]
    path = MODEL_PATHS.get(model_name)
    if model_name == 'Autoencoder':
        model = ae.Autoencoder(input_dim=X_train_scaled.shape[1])
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        loaded_models[model_name] = model
        return model
    if model_name == 'Transformer':
        model = TransformerClassifier(input_dim=X_train_scaled.shape[1])
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        loaded_models[model_name] = model
        return model
    if path and os.path.exists(path):
        model = joblib.load(path)
        loaded_models[model_name] = model
        return model
    return None


def predict(model, model_name, x, y, loader = None, class_loader = None):
    """
    Get predictions for a given model and input data.

    :param model: The model to use for predictions
    :param model_name: The name of the model
    :param x: The features to predict
    :param y: The labels for the features
    :param loader: The data loader to use for the input data (optional)
    :param class_loader: The classification data loader to use for the input data (optional)
    :return: Predictions (array of floats)
    """
    if loader is None or class_loader is None:
        x_loader = data_loader.create_dataloader(x, batch_size=1, shuffle=False, use_gpu=torch.cuda.is_available())
        x_class_loader = data_loader.create_classification_dataloader(x, y, batch_size=1, shuffle=False)
    else:
        x_loader = loader
        x_class_loader = class_loader
    if model_name == 'Autoencoder':
        reconstruction_error = ae.get_reconstruction_error(model, x_loader, device)
        # Set a threshold for anomaly detection
        threshold = np.percentile(reconstruction_error, 95)
        # Get predictions based on the threshold
        pred_ae = np.where(reconstruction_error > threshold, 1, 0)
        return pred_ae
    if model_name == 'Transformer':
        model.eval()
        all_labels_transformer = []
        all_probs_transformer = []
        with torch.no_grad():
            for inputs, labels in x_class_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                probs = torch.sigmoid(outputs)

                all_labels_transformer.extend(labels.cpu().numpy())
                all_probs_transformer.extend(probs.cpu().numpy())
        # Determine optimal threshold
        optimal_threshold, best_f1 = get_optimal_threshold(np.array(all_labels_transformer), np.array(all_probs_transformer))
        preds_transformer = (np.array(all_probs_transformer) >= optimal_threshold).astype(int)
        return preds_transformer
    predictions = model.predict(x)
    predictions = np.where(predictions == 1, 0, 1)  # Convert 1 (normal) to 0 and -1 (anomaly) to 1
    return predictions


def generate_individual_plots(y_test, predictions, model_name):
    """
    Generate and save confusion matrix for the given model.
    """
    cm_path = os.path.join(PLOT_DIR, f'{model_name}_cm.png')
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=["Non-Fraud", "Fraud"], values_format='d')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(cm_path)
    plt.close()
    return cm_path


def generate_combined_pr_roc_curves(y_test, model_preds):
    """
    Generate and save combined precision-recall and ROC curves for all models.
    """
    pr_path = os.path.join(PLOT_DIR, 'combined_pr.png')
    roc_path = os.path.join(PLOT_DIR, 'combined_roc.png')

    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    for model_name, predictions in model_preds.items():
        precision, recall, _ = precision_recall_curve(y_test, predictions)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, marker='.', label=f'{model_name} AUPRC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig(pr_path)
    plt.close()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    for model_name, predictions in model_preds.items():
        fpr, tpr, _ = roc_curve(y_test, predictions)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} AUC = {roc_auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="best")
    plt.savefig(roc_path)
    plt.close()

    return pr_path, roc_path


@app.route('/')
def index():
    return render_template('index.html', models=list(MODEL_PATHS.keys()))


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Evaluate the selected model on the test dataset and display the results.
    """
    selected_models = request.form.getlist('models')
    model_preds = {}
    cm_paths = {}
    classification_reports = {}

    for model_name in selected_models:
        model = load_model(model_name)
        if model:
            predictions = predict(model, model_name, X_test_scaled, Y_test, test_loader, class_test_loader)
            model_preds[model_name] = predictions
            cm_paths[model_name] = generate_individual_plots(Y_test, predictions, model_name)
            report_dict = classification_report(Y_test, predictions, output_dict=True, zero_division=0)
            classification_reports[model_name] = report_dict

    pr_path, roc_path = generate_combined_pr_roc_curves(Y_test, model_preds)

    return render_template(
        'results.html',
        cm_paths=cm_paths,
        classification_reports=classification_reports,
        pr_path=pr_path,
        roc_path=roc_path
    )


@app.route('/generate_live_data', methods=['POST'])
def generate_live_data():
    """
    Generate a new data point and run predictions on the selected models.
    """
    synthetic_data = data_generator.sample(1)
    # Replace nan values with 0.0
    synthetic_data = synthetic_data.fillna(0.0)
    synthetic_x = synthetic_data.drop(columns=[label_column_name])
    original_class = synthetic_data[label_column_name].values[0]
    synthetic_x = synthetic_x.to_numpy()
    original_class: int = round(original_class) if not math.isnan(original_class) else 0
    y = Series([original_class])
    selected_models = request.json.get('selected_models')

    predictions = {}
    for model_name in selected_models:
        model = load_model(model_name)
        if model:
            pred = predict(model, model_name, x=synthetic_x, y=y)
            predictions[model_name] = int(pred[0]) if not math.isnan(pred[0]) else 0
    return jsonify({
        'original_class': original_class,
        'predictions': predictions
    })


@app.route('/live_monitor')
def live_monitor():
    return render_template('live_monitor.html', models=list(MODEL_PATHS.keys()))


if __name__ == '__main__':
    app.run(debug=True)