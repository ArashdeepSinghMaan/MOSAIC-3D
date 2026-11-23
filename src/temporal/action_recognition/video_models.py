"""RGB video backbones (VideoMAE / Video Swin) via HuggingFace transformers."""
from __future__ import annotations

import torch.nn as nn


class VideoMAEClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: str = "MCG-NJU/videomae-base-finetuned-kinetics"):
        super().__init__()
        from transformers import VideoMAEForVideoClassification
        self.model = VideoMAEForVideoClassification.from_pretrained(
            pretrained, num_labels=num_classes, ignore_mismatched_sizes=True)

    def forward(self, pixel_values):  # (B, T, 3, H, W)
        return self.model(pixel_values=pixel_values).logits
