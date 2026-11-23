import numpy as np

from src.evaluation import accuracy, macro_f1, mpjpe, pa_mpjpe, segment_map
from src.evaluation.detection_metrics import average_precision


def test_mpjpe_and_pa():
    gt = np.random.default_rng(0).normal(size=(17, 3))
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    pred = 2.0 * gt @ R.T + 5
    assert mpjpe(gt, gt) == 0 and mpjpe(pred, gt) > 1
    assert pa_mpjpe(pred, gt) < 1e-8


def test_classification_metrics():
    assert accuracy([1, 2, 3], [1, 2, 0]) == 2 / 3
    assert macro_f1([0, 1], [0, 1]) == 1.0


def test_ap_perfect():
    gts = {0: [np.array([0, 0, 10, 10])]}
    assert average_precision([(0, 0.9, np.array([0, 0, 10, 10]))], gts) == 1.0


def test_segment_map():
    g = [{"label": 1, "start": 0, "end": 2}]
    assert segment_map([{"label": 1, "start": 0.1, "end": 2, "score": 1}], g) == 1.0
