import torch
import torch.nn as nn

from src.core.registry import ACTION_MODELS


@ACTION_MODELS.register("transformer")
class TemporalTransformer(nn.Module):
    """Frame tokens (skeleton + optional hand/object/HOI features) -> transformer encoder."""

    def __init__(self, in_dim: int = 51, d_model: int = 256, heads: int = 8, layers: int = 4,
                 num_classes: int = 10, max_len: int = 512, dropout: float = 0.1,
                 per_frame: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True,
                                         norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, num_classes)
        self.per_frame = per_frame

    def forward(self, x, mask=None):  # x: (B, T, D) or (B, T, J, 3)
        if x.dim() == 4:
            x = x.flatten(2)
        B, T, _ = x.shape
        h = torch.cat([self.cls.expand(B, -1, -1), self.proj(x)], 1) + self.pos[:, :T + 1]
        pad = None if mask is None else torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~mask], 1)
        h = self.encoder(h, src_key_padding_mask=pad)
        return self.head(h[:, 1:]) if self.per_frame else self.head(h[:, 0])
