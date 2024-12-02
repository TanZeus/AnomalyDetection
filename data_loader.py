import os
from datetime import datetime

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


def process_dates(df):
    """
    Process the 'trans_date_trans_time' column (convert date to useful numerical features)
    """
    # We'll extract month, day, hour minute second from the date to get more usable information
    # Format: YYYY-MM-DD HH:MM:SS
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S')
    df['month'] = df['trans_date_trans_time'].dt.month
    df['day'] = df['trans_date_trans_time'].dt.day
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['minute'] = df['trans_date_trans_time'].dt.minute
    df['second'] = df['trans_date_trans_time'].dt.second
    df = df.drop(columns=['trans_date_trans_time'])  # Drop the original date column after processing
    return df


def process_dob(df):
    """
    Process the 'dob' column (convert date of birth to useful numerical features)
    """
    # We'll extract year, month and day from the date to get more usable information
    # Format: YYYY-MM-DD
    df['dob'] = pd.to_datetime(df['dob'], format='%Y-%m-%d')
    df['dob_year'] = df['dob'].dt.year
    df['dob_month'] = df['dob'].dt.month
    df = df.drop(columns=['dob'])  # Drop the original date of birth column after processing
    return df


def load_pointe77_data(drop_string_columns=True, limit=None, path='data/pointe77/credit-card-transaction/'):
    """
    Load the 'credit-card-transaction' dataset from 'pointe77'

    :param drop_string_columns: Drop columns that have string values
    :param limit: Limit the number of rows to speed up training during development
    :param path: Path to the dataset files
    """
    # Load the train and test datasets
    start_time = datetime.now()
    print("Loading datasets...")
    train_path = path + 'credit_card_transaction_train.csv'
    test_path = path + 'credit_card_transaction_test.csv'

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    if limit:
        # Limit training/testing data, to speed up training during development
        train_data = train_data.head(limit)
        test_data = test_data.head(limit)

    # Drop unnecessary columns
    if drop_string_columns:
        drop_columns = ['Unnamed: 0', 'cc_num', 'trans_num', 'first', 'last', 'street']
    else:
        drop_columns = ['Unnamed: 0', 'cc_num', 'trans_num']
    train_data = train_data.drop(columns=drop_columns)
    test_data = test_data.drop(columns=drop_columns)

    train_data = process_dates(train_data)
    train_data = process_dob(train_data)

    test_data = process_dates(test_data)
    test_data = process_dob(test_data)

    # Encode categorical columns
    categorical_columns = ['merchant', 'category', 'gender', 'city', 'state', 'job']

    for column in categorical_columns:
        # Combine train and test columns to ensure consistency of encoding
        combined_data = pd.concat([train_data[column], test_data[column]], axis=0)
        # Convert to categorical type and encode as numerical
        train_data[column] = pd.Categorical(train_data[column], categories=combined_data.unique()).codes
        test_data[column] = pd.Categorical(test_data[column], categories=combined_data.unique()).codes

    # Handle missing values (NaNs)
    # Impute numerical features with the mean and categorical features with the most frequent value
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Separate numeric and categorical columns
    numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
    available_categorical_columns = train_data.select_dtypes(
        include=['category', 'object']).columns.tolist()  # Refresh the list

    # Apply imputers to both train and test data if the columns exist
    if numeric_columns:
        train_data[numeric_columns] = num_imputer.fit_transform(train_data[numeric_columns])
        test_data[numeric_columns] = num_imputer.transform(test_data[numeric_columns])

    if available_categorical_columns:  # Only apply imputation if there are categorical columns
        train_data[available_categorical_columns] = cat_imputer.fit_transform(train_data[available_categorical_columns])
        test_data[available_categorical_columns] = cat_imputer.transform(test_data[available_categorical_columns])

    # Separate features and target ('is_fraud' is the target)
    x_train = train_data.drop(columns=['is_fraud'])
    y_train = train_data['is_fraud']

    x_test = test_data.drop(columns=['is_fraud'])
    y_test = test_data['is_fraud']

    # Normalize the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    print(f"Data loading and preprocessing completed in {(datetime.now() - start_time).seconds} seconds")
    return x_train_scaled, x_test_scaled, y_train, y_test


def ctgan_resample(train_data, y_train, target_column, random_state, desired_fraud_ratio=0.01, epochs=100):
    start_time = datetime.now()
    current_fraud_ratio = y_train.sum() / len(train_data)

    numerator = desired_fraud_ratio * len(train_data) - y_train.sum()
    denominator = 1 - desired_fraud_ratio
    num_samples = int(np.ceil(numerator / denominator)) if denominator != 0 else 0

    if num_samples <= 0:
        print(
            f"Desired fraud ratio of {desired_fraud_ratio * 100}% is already achieved or cannot be achieved with the current data."
        )
        augmented_train = train_data
    else:
        ctgan_save_path = f'data/mlg-ulb/credit-card-fraud/ctgan_{epochs}_epochs.pkl'
        print(f"Number of synthetic fraud samples to generate: {num_samples}")
        ctgan_start_time = datetime.now()
        if os.path.exists(ctgan_save_path):
            print(f"Loading CTGAN from {ctgan_save_path}...")
            ctgan = joblib.load(ctgan_save_path)
            print(f"CTGAN loaded in {(datetime.now() - ctgan_start_time).seconds} seconds.")
        else:
            print(f"Training CTGAN...")
            metadata = Metadata.detect_from_dataframe(train_data)
            ctgan = CTGANSynthesizer(metadata=metadata, epochs=epochs, verbose=True)
            ctgan.fit(train_data)
            joblib.dump(ctgan, ctgan_save_path)
            print(f"CTGAN training completed in {(datetime.now() - ctgan_start_time).seconds} seconds.")
        ctgan_start_time = datetime.now()
        # Generate synthetic data
        synthetic_data = ctgan.sample(int(num_samples / current_fraud_ratio / 150))  # TODO: Check if this is too much
        synthetic_frauds = synthetic_data.query(f"{target_column} == 1")

        # Ensure that all synthetic samples are frauds
        num_frauds = synthetic_frauds[target_column].sum()
        print(f"Number of synthetic frauds generated: {num_frauds} in {(datetime.now() - ctgan_start_time).seconds} seconds.")
        if num_frauds > num_samples:
            print(f"Dropping synthetic frauds.")
            fraction = num_samples / num_frauds
            synthetic_frauds = synthetic_frauds.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
            num_frauds = synthetic_frauds[target_column].sum()
            print(f"Number of synthetic frauds after dropping: {num_frauds}")

        # Combine synthetic frauds with the original training data
        augmented_train = pd.concat([train_data, synthetic_frauds], ignore_index=True)

        # Shuffle the augmented training data
        augmented_train = augmented_train.sample(frac=1, random_state=random_state).reset_index(drop=True)

        print(
            f"Augmented training set size: {len(augmented_train)} samples, with {augmented_train[target_column].sum()} " +
            f"fraud cases ({100 * augmented_train[target_column].sum() / len(augmented_train):.2f}%)."
        )
        print(f"Data generation completed in {(datetime.now() - start_time).seconds} seconds.")

    # Separate features and target after augmentation
    x_train = augmented_train.drop(columns=[target_column])
    y_train = augmented_train[target_column]

    return x_train, y_train


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


def load_mlg_ulb_data(csv_path='data/mlg-ulb/credit-card-fraud/creditcard.csv', test_size=0.25, resampling=None, random_state=0):
    """
    Load and preprocess the "Credit Card Fraud Detection" dataset from mlg-ulb

    :param csv_path: Path to the credit card dataset CSV file
    :param test_size: Proportion of the dataset to reserve for testing
    :param resampling: Resampling method to use (default is None, which means no resampling), options are 'smote' and 'ctgan'
    :return: Preprocessed training and test sets (scaled), along with labels
    :param random_state: The random state to use for resampling
    """
    # Load the dataset
    start_time = datetime.now()
    print("Loading the credit card dataset...")

    data = pd.read_csv(csv_path)

    # Perform time-based split based on the 'Time' feature, to mirror real-world data availability
    data = data.sort_values(by='Time').reset_index(drop=True)
    split_index = int(len(data) * (1 - test_size))

    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    target_column = 'Class'
    # Separate features and target ('Class' is the target for fraud detection)
    x_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]

    x_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    if resampling == 'smote':
        print("Applying SMOTE-ENN resampling to the training set...")
        smote_enn = SMOTEENN(random_state=random_state, sampling_strategy=0.015)
        x_train, y_train = smote_enn.fit_resample(x_train, y_train)
        print(f"SMOTE-ENN applied: Resampled training set size: {len(x_train)} samples, original training set size: {len(train_data)}")
        print(f"Resampling completed in {(datetime.now() - start_time).seconds} seconds.")

    if resampling == 'ctgan':
        print("Applying CTGAN-based data generation to the training set...")
        x_train, y_train = ctgan_resample(train_data, y_train, target_column, random_state)

    # Normalize the features (standardize to zero mean and unit variance)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Report distribution of anomalies in train and test sets
    train_fraud_count = y_train.sum()
    test_fraud_count = y_test.sum()

    print(f"Data loaded and processed in {(datetime.now() - start_time).seconds} seconds.")
    print(
        f"Training set: {len(x_train)} samples, with {train_fraud_count} fraud cases ({100 * train_fraud_count / len(y_train):.4f}%)."
    )
    print(
        f"Test set: {len(x_test)} samples, with {test_fraud_count} fraud cases ({100 * test_fraud_count / len(y_test):.4f}%)."
    )

    return x_train_scaled, x_test_scaled, y_train, y_test

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


def load_bot_detection_data(data_path="data/bot-detection/bot.txt", random_state=0, test_size=0.2):
    """
    Load and preprocess the "Bot Detection" dataset

    :param data_path: Path to the data file
    :param random_state: The random state to use
    :param test_size: Proportion of the dataset to reserve for testing
    :return: Preprocessed training and test sets (scaled), along with labels
    """
    start_time = datetime.now()
    print("Loading Bot Detection dataset...")
    # Dataset is in the format: TIMESTAMP IP_ADDRESS IP_COUNTRY QUERY_TYPE SERVER_RESPONSE_CODE URL DOMAIN_ID AGENT_ID BOT_FLAG
    # Where BOT_FLAG is + for bot and doesn't exist for non-bot
    df = pd.read_csv(data_path, sep="\t", header=None, names= [
            "timestamp", "ip_address", "ip_country", "query_type", "server_response_code",
            "url", "domain_id", "agent_id", "bot_flag"
        ]
    )

    # Convert timestamp to 6 columns: year, month, day, hour, minute, second as integers
    # Timestamp is in the format: YYYY-MM-DD HH:MM:SS, so it needs to be split with the space
    df["date"] = df["timestamp"].str.split(" ").str[0]
    df["time"] = df["timestamp"].str.split(" ").str[1]
    df["split"] = df["date"].str.split("-")
    df["year"] = df["split"].str[0]
    df["month"] = df["split"].str[1]
    df["day"] = df["split"].str[2]
    df["split"] = df["time"].str.split(":")
    df["hour"] = df["split"].str[0]
    df["minute"] = df["split"].str[1]
    df["second"] = df["split"].str[2]

    df.drop(columns=["timestamp", "date", "time", "split"], inplace=True)

    # Split IP address into four parts
    df[["ip_part1", "ip_part2", "ip_part3", "ip_part4"]] = df["ip_address"].str.split(".", expand=True).astype(int)
    df.drop(columns=["ip_address"], inplace=True)

    # Encode categorical features (e.g., country, query type)
    le_country = LabelEncoder() # TODO: Make this independent of the dataset
    df["ip_country"] = le_country.fit_transform(df["ip_country"])

    le_query = LabelEncoder() # TODO: Make this independent of the dataset
    df["query_type"] = le_query.fit_transform(df["query_type"])

    # Process the 'url' column using hashing vectorizer
    df["url"] = df["url"].fillna("") # Replace NaN values with empty strings
    vectorizer = HashingVectorizer(n_features=20, alternate_sign=False)
    url_vectors = pd.DataFrame(vectorizer.fit_transform(df["url"]).toarray())
    url_vectors.columns = [f"url_feat_{i}" for i in range(url_vectors.shape[1])]
    df = pd.concat([df.reset_index(drop=True), url_vectors], axis=1)
    df.drop(columns=["url"], inplace=True)

    # Convert bot_flag into binary labels as int (it can be + or NaN)
    df["bot_flag"] = df["bot_flag"].apply(lambda it: 1 if it == "+" else 0)

    # Split features and labels
    x = df.drop(columns=["bot_flag", "agent_id"])  # Drop agent_id to avoid overfitting
    y = df["bot_flag"]

    # Scale numerical features
    scaler = MinMaxScaler()
    x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Dataset loaded and preprocessed in {(datetime.now() - start_time).seconds} seconds.")
    return x_train, x_test, y_train, y_test, le_country, le_query





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
