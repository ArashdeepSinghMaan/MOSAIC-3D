import numpy as np


def reprojection_rmse(P_list, X: np.ndarray, obs: np.ndarray, valid: np.ndarray | None = None) -> float:
    """X: (N,3); obs: (V,N,2); valid: (V,N)."""
    errs = []
    for v, P in enumerate(P_list):
        x = (P @ np.c_[X, np.ones(len(X))].T).T
        uv = x[:, :2] / x[:, 2:3]
        e = np.linalg.norm(uv - obs[v], axis=1)
        errs.append(e if valid is None else e[valid[v]])
    e = np.concatenate(errs)
    e = e[~np.isnan(e)]
    return float(np.sqrt((e ** 2).mean())) if len(e) else float("nan")


def temporal_jitter(seq: np.ndarray, fps: float) -> float:
    """Mean magnitude of 3D acceleration (m/s^2) over a (T,J,3) sequence."""
    acc = np.diff(seq, n=2, axis=0) * fps ** 2
    a = np.linalg.norm(acc, axis=-1)
    return float(np.nanmean(a))


def position_error_3d(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.nanmean(np.linalg.norm(pred - gt, axis=-1)))
