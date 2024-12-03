import os
from datetime import datetime

from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP

import joblib
import numpy as np
import pandas as pd
import torch
from sdv.metadata import Metadata
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from imblearn.combine import SMOTEENN
from sdv.single_table import CTGANSynthesizer


def get_ctgan_generator(x_train, y_train, epochs=100, model_name='cic-unsw-nb15'):
    """
    Get a CTGAN generator for synthetic data generation.

    :param x_train: Training features
    :param y_train: Training labels
    :param epochs: Number of epochs for the CTGAN model
    :param model_name: Name of the dataset
    :return: CTGAN generator
    """
    start_time = datetime.now()
    model_path = f'models/generators/{model_name}/ctgan_{epochs}.pkl'
    if os.path.exists(model_path):
        print(f"\nLoading CTGAN from {model_path}...")
        ctgan = joblib.load(model_path)
        print(f"CTGAN loaded in {(datetime.now() - start_time).seconds} seconds.")
        return ctgan
    # Convert the data to a Pandas DataFrame
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    # Add the y_train label column to x_train
    train_data = pd.concat([x_train, y_train], axis=1)
    metadata = Metadata.detect_from_dataframe(train_data)

    print(f"Starting CTGAN Synthesizer for {epochs} Epochs...")

    ctgan = CTGANSynthesizer(metadata=metadata, epochs=epochs, verbose=True, enforce_rounding=True, enforce_min_max_values=True)
    ctgan.fit(train_data)
    joblib.dump(ctgan, model_path)
    print(f"CTGAN training completed in {(datetime.now() - start_time).seconds} seconds, weights saved to {model_path}")
    return ctgan

cic_unsw_label_name = 'Label'  # TODO: Move this to a config file

def load_cic_unsw_data(data_path="data/cic-unsw-nb15/Data.csv", labels_path="data/cic-unsw-nb15/Label.csv", binary=True, test_size=0.2, random_state=0):
    """
    Load and preprocess the CIC-UNSW-NB15 dataset

    :param data_path: Path to the data CSV file
    :param labels_path: Path to the labels CSV file
    :param binary: Whether to convert labels to binary (benign/malicious) or not
    :param test_size: Proportion of the dataset to reserve for testing
    :param random_state: The random state to use for splitting the data into train and test sets
    :return: Preprocessed training and test sets (scaled), along with labels
    """
    start_time = datetime.now()
    print("Loading CIC-UNSW-NB15 dataset...")
    # Load the data and labels
    data = pd.read_csv(data_path)
    labels = pd.read_csv(labels_path)

    # Merge data and labels
    data[cic_unsw_label_name] = labels[cic_unsw_label_name]

    if binary:
        # Convert all non-benign (0) labels to 1 for binary classification
        data[cic_unsw_label_name] = data[cic_unsw_label_name].apply(lambda it: 0 if it == 0 else 1)

    # Split into features and target
    x = data.drop(columns=[cic_unsw_label_name])
    y = data[cic_unsw_label_name]

    # Normalize the features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Stratified train-test split to keep the class distribution
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    x_train, x_test, y_train, y_test = None, None, None, None
    for train_index, test_index in sss.split(x_scaled, y):
        x_train, x_test = x_scaled[train_index], x_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

    if binary:
        print(f"Train data: {len(x_train) / (len(x_train) + len(x_test))*100:.2f}%, train anomaly: {sum(y_train) / len(y_train) * 100:.2f}%, test anomaly: {sum(y_test) / len(y_test) * 100:.2f}%")
    else:
        # Format to percentage, with 2 decimal places
        print(f"Train data%: {len(x_train) / (len(x_train) + len(x_test))*100:.2f}%")

    print(f"Data loaded and processed in {(datetime.now() - start_time).seconds} seconds")

    return x_train, x_test, y_train, y_test


def ext_features_pcap(pcap_file):
    packets = rdpcap(pcap_file)
    features = []

    for pkt in packets:
        if IP in pkt:
            ip_layer = pkt[IP]
            transport_layer = pkt[TCP] if TCP in pkt else pkt[UDP] if UDP in pkt else None

            feature = {
                'src_ip': ip_layer.src,
                'dst_ip': ip_layer.dst,
                'src_port': transport_layer.sport if transport_layer else None,
                'dst_port': transport_layer.dport if transport_layer else None,
                'protocol': ip_layer.proto,
                'length': len(pkt)
                # Add more features here
                # like time, flags, etc.
            }

            features.append(feature)

    return pd.DataFrame(features)


def load_pcap_data(pcap_files, labels, binary=True, test_size=0.2, random_state=0):
    """
    Load and preprocess data from pcap files

    :param pcap_files: List of paths to pcap files
    :param labels: Corresponding labels for each pcap file
    :param binary: Whether to convert labels to binary (benign/malicious) or not
    :param test_size: Proportion of the dataset to reserve for testing
    :param random_state: The random state to use for splitting the data into train and test sets
    :return: Preprocessed training and test sets (scaled), along with labels
    """
    start_time = datetime.now()
    print("Loading and processing pcap files...")

    # Extract features from each pcap file and combine them into a single DataFrame
    data_frames = [ext_features_pcap(pcap_file) for pcap_file in pcap_files]
    data = pd.concat(data_frames, ignore_index=True)

    # Add labels
    data['label'] = pd.concat([pd.Series([label] * len(df)) for label, df in zip(labels, data_frames)], ignore_index=True)

    if binary:
        # Convert all non-benign (0) labels to 1 for binary classification
        data['label'] = data['label'].apply(lambda it: 0 if it == 0 else 1)

    # Split into features and target
    x = data.drop(columns=['label'])
    y = data['label']

    # Normalize the features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Stratified train-test split to keep the class distribution
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    x_train, x_test, y_train, y_test = None, None, None, None
    for train_index, test_index in sss.split(x_scaled, y):
        x_train, x_test = x_scaled[train_index], x_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

    if binary:
        print(f"Train data: {len(x_train) / (len(x_train) + len(x_test))*100:.2f}%, train anomaly: {sum(y_train) / len(y_train) * 100:.2f}%, test anomaly: {sum(y_test) / len(y_test) * 100:.2f}%")
    else:
        print(f"Train data%: {len(x_train) / (len(x_train) + len(x_test))*100:.2f}%")

    print(f"Data loaded and processed in {(datetime.now() - start_time).seconds} seconds")

    return x_train, x_test, y_train, y_test



def create_dataloader(x, batch_size=64, shuffle=True, use_gpu=False):
    """
    Convert NumPy array to PyTorch tensor and then to DataLoader

    :param x: Features
    :param batch_size: Batch size for the DataLoader
    :param shuffle: Whether to shuffle the data (Should be true for Train and False for Test)
    :param use_gpu: Whether to use GPU for training
    :return: DataLoader
    """
    x_tensor = torch.tensor(x, dtype=torch.float32)
    dataset = TensorDataset(x_tensor)
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=use_gpu, pin_memory_device=device)
    return dataloader


def create_dataloaders(x_train, x_test, batch_size=64, use_gpu=False):
    """
    Convert NumPy arrays to PyTorch tensors and then to DataLoaders
    """
    train_loader = create_dataloader(x_train, batch_size, shuffle=True, use_gpu=use_gpu)
    test_loader = create_dataloader(x_test, batch_size, shuffle=False, use_gpu=use_gpu)
    return train_loader, test_loader


def create_classification_dataloader(x, y: pd.Series, batch_size=64, shuffle=True):
    """
    Convert NumPy arrays to PyTorch tensors and then to DataLoaders for classification.

    :param x: Features
    :param y: Labels
    :param batch_size: Batch size for the DataLoader
    :param shuffle: Whether to shuffle the data (Should be true for Train and False for Test)
    :return: DataLoader
    """
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)  # Assuming y is a Pandas Series
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def create_classification_dataloaders(x_train, x_test, y_train, y_test, batch_size=64):
    """
    Convert NumPy arrays to PyTorch tensors and then to DataLoaders for classification.

    :param x_train: Training features
    :param x_test: Test features
    :param y_train: Training labels
    :param y_test: Test labels
    :param batch_size: Batch size for DataLoaders
    :return: Training and test DataLoaders
    """
    train_loader = create_classification_dataloader(x_train, y_train, batch_size, shuffle=True)
    test_loader = create_classification_dataloader(x_test, y_test, batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == '__main__':
    X_train_scaled, X_test_scaled, Y_train, Y_test = load_cic_unsw_data(binary=True)
    data_generator = get_ctgan_generator(X_train_scaled, Y_train, model_name='cic-unsw-nb15')
