from src.core.config import load_config
from src.data.splits import make_recording_splits


def test_config_overrides():
    cfg = load_config("configs/pipeline/mvp.yaml", ["reconstruction.method=linear", "new.key=3"])
    assert cfg["reconstruction"]["method"] == "linear" and cfg["new"]["key"] == 3


def test_subject_independent_splits():
    recs = [{"recording_id": f"r{i}", "subject_id": f"s{i % 5}"} for i in range(20)]
    s = make_recording_splits(recs)
    subj = {k: {f"s{int(r[1:]) % 5}" for r in v} for k, v in s.items()}
    assert not (subj["train"] & subj["test"]) and not (subj["train"] & subj["val"])
    assert sum(len(v) for v in s.values()) == 20
