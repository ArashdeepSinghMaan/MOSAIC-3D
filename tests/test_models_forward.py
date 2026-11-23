import pytest

torch = pytest.importorskip("torch")


def test_action_models_forward():
    from src.temporal.action_recognition import LSTMActionClassifier, STGCN, TemporalTransformer
    x = torch.randn(2, 32, 17, 3)
    for m in (LSTMActionClassifier(num_classes=5), STGCN(num_classes=5), TemporalTransformer(num_classes=5)):
        assert m(x).shape == (2, 5)
