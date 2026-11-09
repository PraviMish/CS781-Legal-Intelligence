"""Phase 5 historical inputs, configurations, optimization and ZIP contracts."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from phase1.models import (
    DecoderOnlyTransformer, EfficientGQATransformer, EncoderDecoderTransformer, PrefixLMTransformer,
)
from phase5 import models as phase5_models, training
from phase5.data import CaseDataset, internal_split, make_windows, read_cases, statute_vocabulary, targets
from phase5.metrics import calibrate, metrics
from phase5.submission import decode, read_predictions, validate, write_predictions


def tiny(name):
    if name == "efficient_gqa":
        return EfficientGQATransformer(10, 1280, 16, 14, 4, 2, 32)
    if name == "encoder_decoder":
        return EncoderDecoderTransformer(10, 1024, 640, 16, 1, 1, 2, 32, 0.0)
    cls = PrefixLMTransformer if name == "prefix_lm" else DecoderOnlyTransformer
    return cls(10, 1280, 16, 5, 2, 32, 0.0)


def test_official_records_split_and_multihot(tmp_path):
    path = tmp_path / "cases.json"
    raw = [{"id": 1, "text": "same", "labels": [3, 9, 3]},
           {"id": 2, "text": "same", "labels": [9]},
           {"id": 3, "text": "different", "labels": [0]}]
    path.write_text(json.dumps(raw), encoding="utf-8")
    cases = read_cases(path, labeled=True)
    assert statute_vocabulary(cases) == (0, 3, 9)
    assert targets(cases, (0, 3, 9)).tolist() == [[0, 1, 1], [0, 0, 1], [1, 0, 0]]
    assert internal_split(cases, core_ids=["1", "2"]) == (cases[:2], cases[2:])
    with pytest.raises(ValueError, match="overlapping"):
        internal_split(cases, core_ids=["1"])
    path.write_text(json.dumps(raw + [raw[0]]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read_cases(path, labeled=True)


def test_original_window_positions():
    tokenizer = Tokenizer(models.WordLevel(
        {str(i): i + 4 for i in range(1100)} | {"<unk>": 3}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    text = " ".join(str(i) for i in range(1100))
    windows, lengths, count = make_windows(text, tokenizer)
    assert count == 1100 and lengths.tolist() == [512] * 3
    assert windows[:, 1].tolist() == [4, 299, 594]
    assert windows[:, 0].tolist() == [1] * 3 and windows[:, -1].tolist() == [2] * 3
    short, short_lengths, _ = make_windows("0 1", tokenizer)
    assert short_lengths.tolist() == [4, 0, 0]
    assert short[0, :5].tolist() == [1, 4, 5, 2, 0]


@pytest.mark.parametrize("code", phase5_models.RUNS)
def test_architecture_objective_and_checkpoint(code, tmp_path, monkeypatch):
    monkeypatch.setattr(phase5_models, "build_model", tiny)
    run = phase5_models.RUNS[code]
    path = Path(__file__).resolve().parents[1] / "phase5" / "experiments" / f"{code}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    for field in ("arch", "method", "loss", "lr", "epochs", "student", "teacher"):
        assert getattr(run, field) == config.get(field, False if field == "student" else None)
    foundation = tmp_path / "foundation.pt"
    torch.save({"model_name": run.arch, "model_state": tiny(run.arch).state_dict()}, foundation)
    model = phase5_models.make_model(run, foundation)
    inputs = torch.tensor([[[1, 4, 2, 0], [1, 5, 2, 0], [0, 0, 0, 0]]])
    lengths = torch.tensor([[3, 3, 0]])
    logits = model(inputs, lengths)
    assert logits.shape == (1, 100) and torch.isfinite(logits).all()
    gold = torch.zeros_like(logits)
    gold[0, 4] = 1
    teacher = torch.zeros_like(logits) if run.student else None
    training.objective(run.loss, logits, gold, torch.ones(100), teacher).backward()
    assert model.chunk_head.weight.grad.isfinite().all()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    if run.student:
        assert len(model.backbone.blocks) == 6
    if code == "E05":
        assert all(p.grad is None for p in model.backbone.decoder_blocks.parameters())
    checkpoint = tmp_path / "best.pt"
    torch.save({"config": config, "model": model.state_dict()}, checkpoint)
    restored = phase5_models.restore_model(run, checkpoint, "cpu")
    torch.testing.assert_close(restored(inputs, lengths), model.eval()(inputs, lengths))


def test_calibration_and_submission(tmp_path):
    y = np.asarray([[1, 1], [0, 1], [0, 0]])
    scores = metrics(y, np.asarray([[.8, .9], [.5, .9], [.1, .7]]), .6)
    assert scores["macro_f1"] == pytest.approx(.9)
    assert scores["micro_f1"] == pytest.approx(6 / 7)
    thresholds, report = calibrate(y, np.asarray([[.8, .9], [.5, .9], [.1, .7]]))
    assert np.all(thresholds == thresholds[0]) and report["global_macro_f1"] >= .9
    probs = np.zeros((2, 100))
    probs[0, 3], probs[0, 6], probs[1, 8] = .6, .5, .4
    ids = ["first", "second"]
    rows = decode(ids, probs, np.full(100, .5))
    assert [row["pred"] for row in rows] == [[3, 6], [8]]
    path = tmp_path / "submission.zip"
    write_predictions(path, rows, ids)
    assert validate(read_predictions(path), ids) == 2
    with pytest.raises(ValueError, match="order"):
        validate(rows, ids[::-1])
    with pytest.raises(ValueError, match="Invalid"):
        validate([{"id": "first", "pred": [3, 3]}, rows[1]], ids)


def test_interrupted_optimizer_step_resumes_exactly(tmp_path, monkeypatch):
    windows = np.zeros((17, 3, 512), dtype=np.uint16)
    windows[:, 0, :3] = [1, 4, 2]
    lengths = np.tile([3, 0, 0], (17, 1))
    y = np.zeros((17, 100), dtype=np.float32)
    y[:, 4] = 1
    dataset = CaseDataset(windows, lengths, y)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Sequential(torch.nn.Dropout(.1), torch.nn.Linear(3, 100))

        def forward(self, inputs, lengths):
            return self.head(lengths.float())

    run = replace(phase5_models.RUNS["E02"], epochs=1)

    def execute(folder):
        torch.manual_seed(5)
        return training.fit(Toy(), run, dataset, dataset, tmp_path / folder,
                            torch.device("cpu"), None, 42, 1, "test")

    execute("uninterrupted")
    save = training.atomic_checkpoint

    def stop(data, path):
        save(data, path)
        if data.get("next_batch") == 16:
            raise InterruptedError

    monkeypatch.setattr(training, "atomic_checkpoint", stop)
    with pytest.raises(InterruptedError):
        execute("resumed")
    monkeypatch.setattr(training, "atomic_checkpoint", save)
    execute("resumed")
    a, b = [torch.load(tmp_path / folder / "best.pt", weights_only=True)
            for folder in ("uninterrupted", "resumed")]
    assert a["val_score"] == b["val_score"]
    assert all(torch.equal(value, b["model"][key]) for key, value in a["model"].items())
