import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import torch
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import data_loader
from anomaly import plot_precision_recall, plot_roc

model_name = 'bot-detection' # TODO: Move this to a config file

model_path = f'models/{model_name}/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    X_train_scaled, X_test_scaled, y_train, y_test, le_country, le_query = data_loader.load_bot_detection_data()

    start_time = datetime.now()
    # Train and test anomaly detection models
    # 1. Random Forest
    # Check if the model has already been trained
    RF_model_path = model_path + 'rf.pkl'
    if os.path.exists(RF_model_path):
        print(f"\nLoading Random Forest from {RF_model_path}...")
        rf = joblib.load(RF_model_path)
        print(f"Random Forest loaded in {(datetime.now() - start_time).seconds} seconds")
    else:
        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        rf.fit(X_train_scaled, y_train)
        # Save the model
        joblib.dump(rf, RF_model_path)
        print(f"Random Forest training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
    pred_rf = rf.predict(X_test_scaled)

    start_time = datetime.now()
    # 2. XGBoost
    # Check if the model has already been trained
    XGB_model_path = model_path + 'xgb.pkl'
    if os.path.exists(XGB_model_path):
        print(f"\nLoading XGBoost from {XGB_model_path}...")
        xgb = joblib.load(XGB_model_path)
        print(f"XGBoost loaded in {(datetime.now() - start_time).seconds} seconds")
    else:
        print("\nTraining XGBoost...")
        xgb = xgboost.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=0, use_label_encoder=False, eval_metric="logloss")
        xgb.fit(X_train_scaled, y_train)
        # Save the model
        joblib.dump(xgb, XGB_model_path)
        print(f"XGBoost training completed in {(datetime.now() - start_time).seconds} seconds, predicting...")
    pred_xgb = xgb.predict(X_test_scaled)

    values_format = "d"  # The number format to use for the confusion matrix values
    print("\n--- Evaluation Results ---\n")
    plot_path = f'images/{model_name}/'
    # Isolation Forest evaluation
    print("Random Forest:")
    print(classification_report(y_test, pred_rf))
    cm_rf = ConfusionMatrixDisplay.from_predictions(y_test, pred_rf, display_labels=["Non-Fraud", "Fraud"],
                                                    values_format=values_format)
    plt.title("Random Forest Confusion Matrix")
    plt.savefig(plot_path + 'RFcm.png')

    # XGBoost evaluation
    print("XGBoost:")
    print(classification_report(y_test, pred_xgb))
    cm_xgb = ConfusionMatrixDisplay.from_predictions(y_test, pred_xgb, display_labels=["Non-Fraud", "Fraud"],
                                                    values_format=values_format)
    plt.title("XGBoost Confusion Matrix")
    plt.savefig(plot_path + 'XGBcm.png')

    plt.show()

    # Plot precision-recall and ROC for each model
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_precision_recall(y_test, pred_rf, 'Random Forest')
    plot_precision_recall(y_test, pred_xgb, 'XGBoost')
    plt.legend()
    plt.subplot(1, 2, 2)
    plot_roc(y_test, pred_rf, 'Random Forest')
    plot_roc(y_test, pred_xgb, 'XGBoost')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path + 'PRAndRoc.png')




