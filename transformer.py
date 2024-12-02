from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

from data_loader import load_cic_unsw_data,load_mlg_ulb_data, create_classification_dataloaders

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
        """
        Positional encoding module.

        :param d_model: Embedding dimension
        :param max_len: Maximum number of positions (features)
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, 1, d_model)  # (max_len, 1, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        self.register_buffer('pe', pe)  # (max_len, 1, d_model)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        :param x: Input embeddings (seq_len, batch_size, d_model)
        :return: Positional encoded embeddings
        """
        if x.size(0) > self.pe.size(0):
            raise ValueError(f"Sequence length {x.size(0)} exceeds maximum length {self.pe.size(0)}")

        x = x + self.pe[:x.size(0), :, :]  # (seq_len, batch_size, d_model)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=3, dim_feedforward=128, dropout=0.1):
        """
        Transformer-based classifier for tabular data.

        :param input_dim: Number of input features
        :param d_model: Embedding dimension
        :param nhead: Number of attention heads
        :param num_layers: Number of Transformer encoder layers
        :param dim_feedforward: Dimension of the feedforward network
        :param dropout: Dropout rate
        """
        super(TransformerClassifier, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Linear layer to project each feature to d_model dimensions
        self.feature_embedding = nn.Linear(1, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=input_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Output logits for binary classification
        )

    def forward(self, x):
        """
        Forward pass of the model.

        :param x: Input features (batch_size, input_dim)
        :return: Logits (batch_size,)
        """
        # x shape: (batch_size, input_dim)

        # Reshape x to (batch_size, input_dim, 1) to embed each feature individually
        x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)

        # Apply feature embedding: (batch_size, input_dim, d_model)
        x = self.feature_embedding(x)

        # Permute for Transformer: (input_dim, batch_size, d_model)
        x = x.permute(1, 0, 2)

        # Add positional encoding
        x = self.pos_encoder(x)  # (input_dim, batch_size, d_model)

        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(x)  # (input_dim, batch_size, d_model)

        # Aggregate the Transformer outputs (e.g., take the mean across the sequence)
        transformer_output = transformer_output.mean(dim=0)  # (batch_size, d_model)

        # Classification head
        logits = self.classifier(transformer_output)  # (batch_size, 1)

        return logits.squeeze(1)  # (batch_size,)


def compute_class_weights(y_train):
    """
    Compute class weights based on the training labels.

    :param y_train: Training labels (Pandas Series or NumPy array)
    :return: Tensor of class weights
    """
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    return torch.tensor(class_weights, dtype=torch.float32)


def train_model(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    :param model: The Transformer model
    :param train_loader: DataLoader for training data
    :param optimizer: Optimizer
    :param criterion: Loss function
    :param device: Device to train on ('cuda' or 'cpu')
    :return: Average training loss
    """
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # (batch_size,)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss


def get_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray):
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.linspace(0.1, 0.9, 81):
        preds = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, (np.array(all_probs) >= 0.5).astype(int))
    roc_auc = roc_auc_score(all_labels, all_probs)

    optimal_threshold, best_f1 = get_optimal_threshold(np.array(all_labels), np.array(all_probs))
    preds = (np.array(all_probs) >= optimal_threshold).astype(int)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall_curve, precision_curve)

    metrics = {
        'Loss': avg_loss,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': best_f1,
        'AUC-ROC': roc_auc,
        'AUPRC': pr_auc,
        'Optimal Threshold': optimal_threshold
    }

    return metrics


def train_loop(model, train_loader, test_loader, criterion, save_path, device, epochs=20, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_metric = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        model.eval()
        metrics = evaluate_model(model, test_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(
            f"Test Loss: {metrics['Loss']:.4f} | Accuracy: {metrics['Accuracy']:.4f} | Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} | F1-Score: {metrics['F1-Score']:.4f} | AUC-ROC: {metrics['AUC-ROC']:.4f} | AUPRC: {metrics['AUPRC']:.4f} | Optimal Threshold: {metrics['Optimal Threshold']:.4f}")
        # Save the model if AUC-ROC improves
        if metrics['AUC-ROC'] > best_metric:
            best_metric = metrics['AUC-ROC']
            torch.save(model.state_dict(), save_path)
            print("Model saved.")


def main():
    x_train, x_test, y_train, y_test = load_cic_unsw_data()

    batch_size = 64
    train_loader, test_loader = create_classification_dataloaders(x_train, x_test, y_train, y_test, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Model parameters
    input_dim = x_train.shape[1]
    d_model = 64
    nhead = 8
    num_layers = 3
    dim_feedforward = 128
    dropout = 0.1

    model = TransformerClassifier(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                  dim_feedforward=dim_feedforward, dropout=dropout)
    model = model.to(device)

    class_weights = compute_class_weights(y_train)
    class_weights = class_weights.to(device)
    print(f'Class weights: {class_weights}')

    # Define loss function with class weights
    # pos_weight is set to the ratio of positive to negative samples
    pos_weight = class_weights[1] / class_weights[0]
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training parameters
    save_model_path = 'models/cic-unsw-nb15/transformer.pth'
    start_time = datetime.now()
    train_loop(model, train_loader, test_loader, criterion, save_model_path, device)
    print(f"Training completed in {(datetime.now() - start_time).seconds} seconds.")
    # Load the best model
    model.load_state_dict(torch.load(save_model_path, map_location=device))

    # Final evaluation
    final_metrics = evaluate_model(model, test_loader, criterion, device)
    print("Final Evaluation Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")



#    x_train_scaled, x_test_scaled, y_train, y_test = load_mlg_ulb_data(resampling="smote")
#
#    batch_size = 64
#    train_loader, test_loader = create_classification_dataloaders(x_train_scaled, x_test_scaled, y_train, y_test, batch_size=batch_size)
#
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    print(f'Using device: {device}')
#
#    # Model parameters
#    input_dim = x_train_scaled.shape[1]
#    d_model = 64
#    nhead = 8
#    num_layers = 3
#    dim_feedforward = 128
#    dropout = 0.1
#
#    model = TransformerClassifier(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
#                                  dim_feedforward=dim_feedforward, dropout=dropout)
#    model = model.to(device)
#
#    class_weights = compute_class_weights(y_train)
#    class_weights = class_weights.to(device)
#    print(f'Class weights: {class_weights}')
#
#    # Define loss function with class weights
#    # pos_weight is set to the ratio of positive to negative samples
#    pos_weight = class_weights[1] / class_weights[0]
#    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#
#    # Training parameters
#    save_model_path = 'models/mlg-ulb/transformer.pth'
#    start_time = datetime.now()
#    train_loop(model, train_loader, test_loader, criterion, save_model_path, device)
#    print(f"Training completed in {(datetime.now() - start_time).seconds} seconds.")
#    # Load the best model
#    model.load_state_dict(torch.load(save_model_path, map_location=device))
#
#    # Final evaluation
#    final_metrics = evaluate_model(model, test_loader, criterion, device)
#    print("Final Evaluation Metrics:")
#    for metric, value in final_metrics.items():
#        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
