import numpy as np


def contact_probability(distance_m: float, d0: float = 0.05, k: float = 120.0) -> float:
    """Soft contact score: ~1 below d0, decays as a logistic in distance."""
    return float(1.0 / (1.0 + np.exp(np.clip(k * (distance_m - d0), -50, 50))))
