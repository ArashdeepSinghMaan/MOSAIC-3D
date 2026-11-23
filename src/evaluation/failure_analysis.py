"""Slice metrics by failure condition (occlusion, blur, low light, sync error, ...)."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import cv2
import numpy as np

FAILURE_TAGS = ["partial_occlusion", "full_occlusion", "sync_error", "poor_calibration", "motion_blur",
                "low_illumination", "fast_hand_motion", "small_object", "similar_people",
                "appearance_change", "tracking_drift", "keypoint_association", "reprojection",
                "domain_shift"]


def blur_score(image: np.ndarray) -> float:
    """Variance of Laplacian — low means blurry."""
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def brightness(image: np.ndarray) -> float:
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean() if image.ndim == 3 else image.mean())


def auto_tags(image: np.ndarray, blur_thresh: float = 100.0, dark_thresh: float = 50.0) -> list[str]:
    tags = []
    if blur_score(image) < blur_thresh:
        tags.append("motion_blur")
    if brightness(image) < dark_thresh:
        tags.append("low_illumination")
    return tags


def sliced_report(samples: list[dict], metric: Callable[[list[dict]], float]) -> dict[str, float]:
    """samples: [{'tags': [...], ...}] -> metric per tag, plus 'all'."""
    by_tag = defaultdict(list)
    for s in samples:
        by_tag["all"].append(s)
        for t in s.get("tags", []):
            by_tag[t].append(s)
    return {t: metric(v) for t, v in by_tag.items()}
