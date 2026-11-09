"""Offline regressions for the Phase 4 release."""

import numpy as np
import pytest
import torch
from phase4.data import (
    SpecialTokens,
    Split,
    make_splits,
    read_split,
    save_split,
    windows,
)
from phase4.gqa import CJPEGQAClassifier
from phase4.metrics import metrics, select_blend, select_threshold
from phase4.submission import read_submission, write_submission
from phase4.train import train
from phase4.variants import (
    E02HierarchicalHead,
    E03LoRMAPlusLinear,
    E07AdaptiveLinear,
    allocate_ranks,
)
from torch import nn


def test_split_and_windows(tmp_path):
    train_rows = [{"id": str(i), "text": f"Judgment {i}", "label": i % 2} for i in range(30)]
    train_rows.insert(0, {"id": "0", "text": "Judgment", "label": 0})
    test = [{"id": "test", "text": "Held-out test"}]
    dev = [
        {"id": "safe", "text": "Development", "label": 1},
        {"id": "test", "text": "Different text", "label": 0},
        {"id": "copy", "text": "  HELD-OUT  TEST  ", "label": 1},
    ]
    split = make_splits(train_rows, dev, test)
    assert [row["id"] for row in split["dev_safe"]] == ["safe"]
    assert sum(len(split[name]) for name in ("train_core", "val_internal")) == 30

    special = SpecialTokens(0, 1, 2)
    tokens = list(range(3, 19))
    tail, length = windows(tokens, special, width=8)
    hierarchical, lengths = windows(tokens, special, width=8, strategy="hierarchical")
    assert tail[0].tolist() == [1, *tokens[-6:], 2]
    assert np.array_equal(tail[0], hierarchical[(lengths > 0).sum() - 1])
    packed = Split(["id"], np.array([1]), tail[None], length[None])
    save_split(tmp_path / "tail", packed)
    restored = read_split(tmp_path / "tail")
    assert restored.ids == ["id"]
    assert np.array_equal(restored.tokens, packed.tokens)


def test_threshold_ensemble_and_submission(tmp_path):
    labels = np.array([0, 1, 0, 1])
    base = np.array([0.12, 0.79, 0.24, 0.80], dtype=np.float32)
    other = np.array([0.16, 0.75, 0.28, 0.70], dtype=np.float32)
    assert metrics(labels, base)["macro_f1"] == 1
    assert select_threshold(labels, base, [0.4, 0.5, 0.6], tie_near_half=True) == 0.5
    assert select_blend(labels, base, other) == (0.55, 0.26, 1.0)
    ids = ["c", "a", "b", "d"]
    zip_path = tmp_path / "submission.zip"
    write_submission(zip_path, ids, base, 0.51)
    assert [row["pred"] for row in read_submission(zip_path, ids)] == labels.tolist()
    with pytest.raises(ValueError, match="differ"):
        read_submission(zip_path, ids[:-1])


def test_heads_and_adapters():
    head = E02HierarchicalHead(hidden_dim=8, num_chunks=3)
    features, logits = torch.randn(2, 3, 8), torch.randn(2, 3, 2)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    torch.testing.assert_close(head(features, logits, mask), logits[[0, 1], [1, 0]])
    base = nn.Linear(8, 8)
    x = torch.randn(3, 8)
    torch.testing.assert_close(E03LoRMAPlusLinear(base)(x), base(x))
    adaptive = E07AdaptiveLinear(base, rank=4)
    torch.testing.assert_close(adaptive(x), base(x))
    model = type("AdapterHolder", (), {"adapters": [adaptive]})()
    assert allocate_ranks(model, 2) == 2
    assert allocate_ranks(model, 4) == 2


def test_e01_training_resume(tmp_path):
    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10, 4)

        def hidden_states(self, ids):
            return self.embedding(ids), None

    model = CJPEGQAClassifier(TinyBackbone(), hidden_dim=4, dropout=0.0)
    tokens = np.array([[[1, 3, 2]], [[1, 4, 2]], [[1, 5, 2]]], dtype=np.uint16)
    split = Split(["a", "b", "c"], np.array([0, 1, 0]), tokens, np.full((3, 1), 3))
    config = {
        "seed": 42,
        "micro_batch_size": 1,
        "gradient_accumulation": 2,
        "max_epochs": 1,
        "backbone_lr": 2e-5,
        "head_lr": 3e-4,
        "weight_decay": 0.01,
        "checkpoint_every_steps": 1,
        "early_stopping_patience": 2,
    }
    device = torch.device("cpu")
    best = train(model, "E01", config, split, split, tmp_path, device, "fp32", identity=["test"])
    checkpoint = torch.load(best, weights_only=True)
    assert checkpoint["global_step"] == 2 and checkpoint["epoch"] == 0
    model2 = CJPEGQAClassifier(TinyBackbone(), hidden_dim=4, dropout=0.0)
    train(model2, "E01", config, split, split, tmp_path, device, "fp32", identity=["test"])
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(model2.state_dict()[key], tensor)
