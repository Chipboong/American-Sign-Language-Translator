import torch
from torch.utils.data import DataLoader
import pandas as pd
from src.data.data_loader import LandmarkDataset
from src.models.models import LSTMClassifier
from src.utils.utils import load_model, accuracy_score
from tqdm import tqdm

# Evaluation configuration
BATCH_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
VAL_CSV = 'src/data/metadata/val.csv'  # Update with your validation CSV path
DATA_DIR = 'dataset/'
MODEL_PATH = 'asl_checkpoint_1.pth'
SEQ_LEN = 384  # match training config

# Load dataset
val_dataset = LandmarkDataset(VAL_CSV, DATA_DIR, seq_len=SEQ_LEN)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = LSTMClassifier(
    input_dim=val_dataset.num_features,
    hidden_dim=128,
    num_layers=2,
    num_classes=val_dataset.num_classes
).to(DEVICE)
model = load_model(model, MODEL_PATH, DEVICE)
model.eval()

# Evaluation loop
def evaluate():
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.append(labels.cpu())
            all_preds.append(predicted.cpu())
    accuracy = 100 * correct / total if total > 0 else 0
    print(f'Validation Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    evaluate()
