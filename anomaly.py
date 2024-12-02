import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from torch import nn

import data_loader
import auto_encoder as ae
from transformer import TransformerClassifier, train_loop, compute_class_weights, get_optimal_threshold

# model_name = 'pointe77'
# model_name = 'mlg-ulb/default'
# model_name = 'mlg-ulb/smote'
# model_name = 'mlg-ulb/ctgan'
model_name = 'cic-unsw-nb15' # TODO: Move these to a config file

model_path = f'models/{model_name}/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_precision_recall(y_tests, pred, model):
    """
    Plot Precision-Recall curves and compute AUPRC.
    """
    precision, recall, _ = precision_recall_curve(y_tests, pred)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, marker='.', label=f'{model} AUPRC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')


def plot_roc(y_tests, pred, model):
    """
    Plot ROC curves
    """
    fpr, tpr, _ = roc_curve(y_tests, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model} AUC = {roc_auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')


if __name__ == '__main__':
    # X_train_scaled, X_test_scaled, y_train, y_test = data_loader.load_pointe77_data()
    # X_train_scaled, X_test_scaled, y_train, y_test = data_loader.load_mlg_ulb_data(resampling=None)
    # X_train_scaled, X_test_scaled, y_train, y_test = data_loader.load_mlg_ulb_data(resampling='smote')
    # X_train_scaled, X_test_scaled, y_train, y_test = data_loader.load_mlg_ulb_data(resampling='ctgan')
    X_train_scaled, X_test_scaled, y_train, y_test = data_loader.load_cic_unsw_data(binary=True)

    train_loader, test_loader = data_loader.create_dataloaders(X_train_scaled, X_test_scaled, use_gpu=torch.cuda.is_available())

    class_train_loader, class_test_loader = data_loader.create_classification_dataloaders(X_train_scaled, X_test_scaled, y_train, y_test)

    # Set the contamination parameter based on the train dataset (percentage of fraud cases)
    contamination = 0.2 if model_name == 'cic-unsw-nb15' else y_train.sum() / len(y_train)
    print(f"Contamination: {contamination}")

    start_time = datetime.now()
    # Train and test anomaly detection models
    # 1. Isolation Forest
    # Check if the model has already been trained
    ISOFOREST_model_path = model_path + 'iso_forest.pkl'
    if os.path.exists(ISOFOREST_model_path):
        print(f"\nLoading Isolation Forest from {ISOFOREST_model_path}...")
        isolation_forest = joblib.load(ISOFOREST_model_path)
        print(f"Isolation Forest loaded in {(datetime.now() - start_time).seconds} seconds")
    else:
        print("\nTraining Isolation Forest...")
        isolation_forest = IsolationForest(contamination=contamination)
        isolation_forest.fit(X_train_scaled)
        # Save the model
        joblib.dump(isolation_forest, ISOFOREST_model_path)
        print(f"Isolation Forest training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
    pred_if = isolation_forest.predict(X_test_scaled)
    pred_if = np.where(pred_if == 1, np.array(0, dtype=pred_if.dtype), np.array(1, dtype=pred_if.dtype))  # Convert 1 (normal) to 0 and -1 (anomaly) to 1

    start_time = datetime.now()
    # 2. Local Outlier Factor
    # Check if the model has already been trained
    LOF_model_path = model_path + 'lof.pkl'
    if os.path.exists(LOF_model_path):
        print(f"\nLoading Local Outlier Factor from {LOF_model_path}...")
        lof = joblib.load(LOF_model_path)
        print(f"LOF loaded in {(datetime.now() - start_time).seconds} seconds")
    else:
        print("\nTraining Local Outlier Factor...")
        lof = LocalOutlierFactor(contamination=contamination, novelty=True, n_neighbors=10)
        lof.fit(X_train_scaled)
        # Save the model
        joblib.dump(lof, LOF_model_path)
        print(f"LOF training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
    pred_lof = lof.predict(X_test_scaled)
    pred_lof = np.where(pred_lof == 1, np.array(0, dtype=pred_if.dtype), np.array(1, dtype=pred_if.dtype))  # Convert 1 (normal) to 0 and -1 (anomaly) to 1

    start_time = datetime.now()
    # 3. One-Class SVM
    # Check if the model has already been trained
    OCSVM_model_path = model_path + 'ocsvm.pkl'
    if os.path.exists(OCSVM_model_path):
        print(f"\nLoading One-Class SVM from {OCSVM_model_path}...")
        ocsvm = joblib.load(OCSVM_model_path)
        print(f"OCSVM loaded in {(datetime.now() - start_time).seconds} seconds")
    else:
        print("\nTraining One-Class SVM...")
        ocsvm = OneClassSVM(nu=0.005, kernel='rbf', gamma='scale')
        ocsvm.fit(X_train_scaled)
        # Save the model
        joblib.dump(ocsvm, OCSVM_model_path)
        print(f"OCSVM training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
    pred_ocsvm = ocsvm.predict(X_test_scaled)
    pred_ocsvm = np.where(pred_ocsvm == 1, np.array(0, dtype=pred_if.dtype), np.array(1, dtype=pred_if.dtype))  # Convert 1 (normal) to 0 and -1 (anomaly) to 1

    start_time = datetime.now()
    # 4. K-Means clustering
    # Check if the model has already been trained
    KMEANS_model_path = model_path + 'kmeans.pkl'
    if os.path.exists(KMEANS_model_path):
        print(f"\nLoading K-Means from {KMEANS_model_path}...")
        kmeans = joblib.load(KMEANS_model_path)
        print(f"K-Means loaded in {(datetime.now() - start_time).seconds} seconds")
    else:
        print("\nTraining K-Means...")
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(X_train_scaled)
        # Save the model
        joblib.dump(kmeans, KMEANS_model_path)
        print(f"K-Means training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
    pred_kmeans = kmeans.predict(X_test_scaled)
    pred_kmeans = np.where(pred_kmeans == 1, np.array(0, dtype=pred_if.dtype), np.array(1, dtype=pred_if.dtype))  # Convert 1 (normal) to 0 and -1 (anomaly) to 1

    start_time = datetime.now()
    # 5. Autoencoder
    # Define the autoencoder model
    autoencoder_model = ae.Autoencoder(input_dim=X_train_scaled.shape[1])
    ae_model_path = model_path + 'autoencoder.pth'

    # Load the trained model if it exists
    if os.path.exists(ae_model_path):
        print(f"\nLoading Autoencoder from {ae_model_path}...")
        autoencoder_model.load_state_dict(torch.load(ae_model_path, map_location=device))
        print(f"Autoencoder model loaded in {(datetime.now() - start_time).seconds} seconds")
    else:
        # Train the model if it's not already saved
        print("\nTraining Autoencoder...")
        autoencoder_model = ae.train_autoencoder(autoencoder_model, train_loader, device, epochs=10, lr=0.001)
        torch.save(autoencoder_model.state_dict(), ae_model_path)
        print(f"Autoencoder model training completed in {(datetime.now() - start_time).seconds} seconds, weights saved to {ae_model_path}")
    autoencoder_model = autoencoder_model.to(device)
    # Get reconstruction error on the test data
    reconstruction_error = ae.get_reconstruction_error(autoencoder_model, test_loader, device)
    # Set a threshold for anomaly detection
    threshold = np.percentile(reconstruction_error, 95)
    # Get predictions based on the threshold
    pred_ae = np.where(reconstruction_error > threshold, 1, 0)

    start_time = datetime.now()
    # 6. Transformer
    # Define the Transformer model
    transformer_model = TransformerClassifier(input_dim=X_train_scaled.shape[1])
    transformer_model = transformer_model.to(device)

    transformer_model_path = model_path + 'transformer.pth'

    # Load the trained Transformer model if exists
    if os.path.exists(transformer_model_path):
        print(f"\nLoading Transformer model from {transformer_model_path}...")
        transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=device))
        print(f"Transformer model loaded in {(datetime.now() - start_time).seconds} seconds")
    else:
        # Train the model if it's not already saved
        print("\nTraining Transformer...")
        class_weights = compute_class_weights(y_train)
        class_weights = class_weights.to(device)
        pos_weight = class_weights[1] / class_weights[0]
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_loop(transformer_model, class_train_loader, class_test_loader, criterion, transformer_model_path, device)
        transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=device))
        print(f"Transformer model training completed in {(datetime.now() - start_time).seconds} seconds, weights saved to {transformer_model_path}")

    # Transformer evaluation
    transformer_model.eval()
    all_labels_transformer = []
    all_probs_transformer = []

    with torch.no_grad():
        for inputs, labels in class_test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = transformer_model(inputs)
            probs = torch.sigmoid(outputs)

            all_labels_transformer.extend(labels.cpu().numpy())
            all_probs_transformer.extend(probs.cpu().numpy())

    # Determine optimal threshold
    optimal_threshold, best_f1 = get_optimal_threshold(np.array(all_labels_transformer), np.array(all_probs_transformer))
    preds_transformer = (np.array(all_probs_transformer) >= optimal_threshold).astype(int)

    print(f"Transformer prediction shape: {preds_transformer.shape}")

    start_time = datetime.now()
    # 7. One-Class Nearest Neighbors
    # Check if the model has already been trained
    OCNN_model_path = model_path + 'ocnn.pkl'
    if os.path.exists(OCNN_model_path):
        print(f"\nLoading One-Class Nearest Neighbors from {OCNN_model_path}...")
        ocnn = joblib.load(OCNN_model_path)
        print(f"One-Class Nearest Neighbors loaded in {(datetime.now() - start_time).seconds} seconds")
    else:
        print("\nTraining One-Class Nearest Neighbors...")
        ocnn = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='minkowski', p=2, n_jobs=-1)
        ocnn.fit(X_train_scaled)
        # Save the model
        joblib.dump(ocnn, OCNN_model_path)
        print(f"One-Class Nearest Neighbors training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
    ocnn_distances, _ = ocnn.kneighbors(X_test_scaled)
    mean_distances = np.mean(ocnn_distances, axis=1)
    ocnn_threshold = np.percentile(mean_distances, 95)
    pred_ocnn = np.where(mean_distances > ocnn_threshold, 1, 0)

    values_format = "d"  # The number format to use for the confusion matrix values
    print("\n--- Evaluation Results ---\n")
    plot_path = f'images/{model_name}/'
    # Isolation Forest evaluation
    print("Isolation Forest:")
    print(classification_report(y_test, pred_if))
    cm_if = ConfusionMatrixDisplay.from_predictions(y_test, pred_if, display_labels=["Non-Fraud", "Fraud"],
                                                    values_format=values_format)
    plt.title("Isolation Forest Confusion Matrix")
    plt.savefig(plot_path + 'IFcm.png')

    # Local Outlier Factor evaluation
    print("Local Outlier Factor:")
    print(classification_report(y_test, pred_lof))
    cm_lof = ConfusionMatrixDisplay.from_predictions(y_test, pred_lof, display_labels=["Non-Fraud", "Fraud"],
                                                     values_format=values_format)
    plt.title("LOF Confusion Matrix")
    plt.savefig(plot_path + 'LOFcm.png')

    # One-Class SVM evaluation
    print("One-Class SVM:")
    print(classification_report(y_test, pred_ocsvm))
    cm_ocsvm = ConfusionMatrixDisplay.from_predictions(y_test, pred_ocsvm, display_labels=["Non-Fraud", "Fraud"],
                                                       values_format=values_format)
    plt.title("One-Class SVM Confusion Matrix")
    plt.savefig(plot_path + 'OCSVMcm.png')

    # K-Means evaluation
    print("K-Means:")
    print(classification_report(y_test, pred_kmeans))
    cm_kmeans = ConfusionMatrixDisplay.from_predictions(y_test, pred_kmeans, display_labels=["Non-Fraud", "Fraud"],
                                                        values_format=values_format)
    plt.title("K-Means Confusion Matrix")
    plt.savefig(plot_path + 'KMcm.png')

    # Autoencoder evaluation
    print("Autoencoder (AE):")
    print(classification_report(y_test, pred_ae))
    cm_ae = ConfusionMatrixDisplay.from_predictions(y_test, pred_ae, display_labels=["Non-Fraud", "Fraud"],
                                                    values_format=values_format)
    plt.title("Autoencoder Confusion Matrix")
    plt.savefig(plot_path + 'AEcm.png')

    # Transformer evaluation
    print("Transformer:")
    print(classification_report(y_test, preds_transformer))
    cm_transformer = ConfusionMatrixDisplay.from_predictions(y_test, preds_transformer, display_labels=["Non-Fraud", "Fraud"],
                                                             values_format=values_format)
    plt.title("Transformer Confusion Matrix")
    plt.savefig(plot_path + 'Transformercm.png')

    # One-Class Nearest Neighbors evaluation
    print("One-Class Nearest Neighbors:")
    print(classification_report(y_test, pred_ocnn))
    cm_ocnn = ConfusionMatrixDisplay.from_predictions(y_test, pred_ocnn, display_labels=["Non-Fraud", "Fraud"],
                                                      values_format=values_format)
    plt.title("One-Class Nearest Neighbors Confusion Matrix")
    plt.savefig(plot_path + 'OCNNcm.png')

    plt.show()

    # Plot precision-recall and ROC for each model
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_precision_recall(y_test, pred_if, 'Isolation Forest')
    plot_precision_recall(y_test, pred_lof, 'LOF')
    plot_precision_recall(y_test, pred_ocsvm, 'One-Class SVM')
    plot_precision_recall(y_test, pred_kmeans, 'K-Means')
    plot_precision_recall(y_test, pred_ae, 'Autoencoder')
    plot_precision_recall(y_test, preds_transformer, 'Transformer')
    plot_precision_recall(y_test, pred_ocnn, 'One-Class Nearest Neighbors')
    plt.legend()
    plt.subplot(1, 2, 2)
    plot_roc(y_test, pred_if, 'Isolation Forest')
    plot_roc(y_test, pred_lof, 'LOF')
    plot_roc(y_test, pred_ocsvm, 'One-Class SVM')
    plot_roc(y_test, pred_kmeans, 'K-Means')
    plot_roc(y_test, pred_ae, 'Autoencoder')
    plot_roc(y_test, preds_transformer, 'Transformer')
    plot_roc(y_test, pred_ocnn, 'One-Class Nearest Neighbors')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path + 'PRAndRoc.png')
    plt.show()
