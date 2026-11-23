import numpy as np

from src.acquisition.synchronization import (FrameSynchronizer, detect_dropped_frames,
                                             estimate_offset_from_signals)


def test_sync_pairs_and_drops():
    s = FrameSynchronizer(["a", "b"], tolerance=0.01)
    img = np.zeros((2, 2))
    for i, t in enumerate([0.0, 0.033, 0.066]):
        s.push("a", i, t, img)
    for i, t in enumerate([0.034, 0.067]):  # b missed the first frame
        s.push("b", i, t, img)
    fs = s.pop()
    assert fs.frame_ids == {"a": 1, "b": 0} and s.dropped["a"] == 1
    assert s.pop().frame_ids == {"a": 2, "b": 1}


def test_detect_dropped():
    ts = np.array([0, 1, 2, 4, 5]) / 30
    assert list(detect_dropped_frames(ts, 30)) == [3]


def test_offset_estimation():
    sig = np.zeros(200); sig[50] = 1
    assert abs(estimate_offset_from_signals(sig, np.roll(sig, 10), 30.0) - (-10 / 30)) < 1e-9
