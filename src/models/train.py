import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
from src.data.data_loader import LandmarkDataset
from src.models.models import LSTMClassifier
from src.utils.utils import set_seed, save_model
from tqdm import tqdm

# Training configuration
BATCH_SIZE = 32 # train 32 samples at a time
EPOCHS = 400 # used 400 epochs
LEARNING_RATE = 4e-3 # used 4e-3
SEQ_LEN = 384 # used 384 for training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use GPU if available

# Paths
TRAIN_CSV = 'src/data/metadata/train.csv'
DATA_DIR = 'dataset/'

# Set seed for reproducibility (important for consistent results)
set_seed(42)

# Load dataset
train_dataset = LandmarkDataset(TRAIN_CSV, DATA_DIR, seq_len=SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, loss, optimizer
model = LSTMClassifier(
    input_dim=train_dataset.num_features,
    hidden_dim=128,
    num_layers=2,
    num_classes=train_dataset.num_classes
).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # used label smoothing
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0 # lower loss means better performance
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss/len(train_loader):.4f}")
    save_model(model, 'asl_checkpoint_1.pth')
    print('Training complete. Model saved as asl_checkpoint_1.pth')

if __name__ == '__main__':
    train()
