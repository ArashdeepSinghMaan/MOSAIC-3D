"""Predict the next action(s) at horizon tau from an observed window."""
import torch
import torch.nn as nn


class ActionAnticipator(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256, horizons=(1.0, 2.0, 5.0)):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, 2, batch_first=True)
        self.heads = nn.ModuleDict({f"{h:g}s": nn.Linear(hidden, num_classes) for h in horizons})

    def forward(self, x):  # (B, T, D) -> {horizon: logits}
        _, h = self.gru(x)
        return {k: head(h[-1]) for k, head in self.heads.items()}

    @staticmethod
    def topk(logits: torch.Tensor, k: int = 5):
        return logits.softmax(-1).topk(k, -1)
