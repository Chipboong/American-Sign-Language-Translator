import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy_score(y_true, y_pred):
    """Compute accuracy score."""
    correct = (y_true == y_pred).sum().item()
    total = y_true.size(0)
    return correct / total if total > 0 else 0


def save_model(model, path):
    """Save PyTorch model to file."""
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    """Load PyTorch model from file."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model
