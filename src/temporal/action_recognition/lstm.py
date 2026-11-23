import torch
import torch.nn as nn

from src.core.registry import ACTION_MODELS


@ACTION_MODELS.register("lstm")
class LSTMActionClassifier(nn.Module):
    """Baseline: (B, T, J, 3) skeleton sequence -> (B, num_classes)."""

    def __init__(self, num_joints: int = 17, in_ch: int = 3, hidden: int = 256, layers: int = 2,
                 num_classes: int = 10, dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(num_joints * in_ch, hidden, layers, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        B, T = x.shape[:2]
        out, _ = self.lstm(x.reshape(B, T, -1))
        return self.head(out.mean(1))
