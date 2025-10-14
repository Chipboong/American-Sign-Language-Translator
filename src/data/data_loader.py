import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import pyarrow.parquet as pq

class LandmarkDataset(Dataset):
    """
    PyTorch Dataset for Google Isolated Sign Language Recognition landmarks.
    Loads samples from train.csv and .parquet files.
    """
    # This function initializes the dataset by reading the CSV file
    # and setting up paths to the .parquet files.
    def __init__(self, csv_path, data_root, label_map_path=None, transform=None, seq_len=300):
        import json
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.transform = transform
        self.samples = self.df[['path', 'sign']].values
        # print(f"Loaded {len(self.samples)} samples from {csv_path}")
        self.seq_len = seq_len
        # Load label_map from JSON file
        if label_map_path is None:
            label_map_path = os.path.join(data_root, 'labels.json')
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        # Infer num_features from the first sample
        first_path, _ = self.samples[0]
        parquet_path = os.path.join(self.data_root, first_path)
        df = pd.read_parquet(parquet_path)
        self.num_features = df[['x', 'y', 'z']].values.shape[1]
        # Number of classes from label_map
        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, sign = self.samples[idx]
        parquet_path = os.path.join(self.data_root, path)
        df = pd.read_parquet(parquet_path)
        features = df[['x', 'y', 'z']].values.astype('float32')
        # Pad or truncate to self.seq_len
        if features.shape[0] < self.seq_len:
            pad_width = self.seq_len - features.shape[0]
            pad = np.zeros((pad_width, features.shape[1]), dtype='float32')
            features = np.concatenate([features, pad], axis=0)
        elif features.shape[0] > self.seq_len:
            features = features[:self.seq_len]
        label = self.label_map[sign]
        if self.transform:
            features = self.transform(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(features), torch.tensor(label)

# Example usage:
# dataset = LandmarkDataset(csv_path='../data/train.csv', data_root='../')
# X, y = dataset[0]
