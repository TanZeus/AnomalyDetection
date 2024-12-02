import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Compress the input to a lower-dimensional representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        # Reconstruct the input from the compressed representation
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(model, train_loader, device, epochs=10, lr=0.001):
    """
    Train the autoencoder model

    :param model: The autoencoder model
    :param train_loader: The training data loader
    :param device: The device to use for training (e.g., 'cuda' or 'cpu')
    :param epochs: The number of epochs to train the model for
    :param lr: The learning rate for the optimizer
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

    print("Autoencoder training completed.")
    return model


def get_reconstruction_error(model, data_loader, device):
    """
    Get the reconstruction error on the test data

    :param model: The trained autoencoder model
    :param data_loader: The test data loader
    :param device: The device to use for testing (e.g., 'cuda' or 'cpu')
    """
    model.eval()
    errors = []
    criterion = nn.MSELoss(reduction='none')  # We need individual losses

    with torch.no_grad():
        for data in data_loader:
            inputs = data[0].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss_per_sample = loss.mean(dim=1).cpu().numpy()  # Mean per sample
            errors.extend(loss_per_sample)

    return np.array(errors)
