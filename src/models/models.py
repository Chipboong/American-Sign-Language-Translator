import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for sign language recognition.
    Input: (batch_size, seq_len, input_dim)
    Output: (batch_size, num_classes)
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            out = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            out = h_n[-1]
        out = self.dropout(out)
        return self.fc(out)

# Notes
# - batach_size: Number of samples in a batch.
# - seq_len: Number of time steps (frames) in each sample.

# - input_dim: Number of input features per time step (e.g., 3 for x,y,z).
# - hidden_dim: Number of features in the hidden state of the LSTM.
# - num_classes: Number of output classes for classification.
# - num_layers: Number of stacked LSTM layers.
# - dropout: Dropout rate for regularization.

# Example usage:
# model = LSTMClassifier(input_dim=3, hidden_dim=128, num_classes=50, num_layers=2, dropout=0.5, bidirectional=True)
# output = model(torch.randn(16, 384, 3))
