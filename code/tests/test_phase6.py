"""Phase 6 preprocessing, checkpoints, objectives, generation and packaging tests."""

import json
from dataclasses import replace
from zipfile import ZipFile

import pytest
import torch
from torch import nn
from tokenizers import Tokenizer, models, pre_tokenizers

from phase1.models import (
    DecoderOnlyTransformer, EfficientGQATransformer, EncoderDecoderTransformer, PrefixLMTransformer,
)
from phase6.data import BOS, EOS, PassageInputs, load_records, split_records
from phase6.decoding import dedupe, evidence_filter, generate_passage, next_token
from phase6.models import configuration, configure_model
from phase6.preparation import SectionInputs, training_group
from phase6.submission import read_submission, write_submission
from phase6.training import distillation_loss, schedule, target_logits, train


def tiny(architecture):
    if architecture == "encoder_decoder":
        return EncoderDecoderTransformer(10, 1024, 640, 16, 1, 1, 2, 32, 0.0)
    if architecture == "efficient_gqa":
        return EfficientGQATransformer(10, 1280, 16, 2, 4, 2, 32)
    if architecture == "prefix_lm":
        return PrefixLMTransformer(10, 1280, 16, 2, 2, 32, 0.0)
    return DecoderOnlyTransformer(10, 1280, 16, 12, 2, 32, 0.0)


@pytest.fixture
def tokenizer():
    tok = Tokenizer(models.WordLevel({
        "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "judgment": 4,
        "reasoning": 5, "dismissed": 6, "Facts": 7, ".": 8, "dated": 9,
    }, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


def test_official_format_and_grouped_split(tmp_path):
    path = tmp_path / "input.json"
    raw = [{"id": 1, "document": " A judgment ", "summary": "Facts."},
           {"id": "2", "document": "A   judgment", "summary": "Facts."},
           {"id": "3", "document": "Another judgment", "summary": "More."}]
    path.write_text(json.dumps(raw), encoding="utf-8")
    rows = load_records(path, labeled=True)
    assert [row["id"] for row in rows] == ["1", "2", "3"]
    assert split_records(rows, ["3"]) == (rows[:2], rows[2:])
    with pytest.raises(ValueError, match="duplicate text"):
        split_records(rows, ["1"])
    path.write_text(json.dumps(raw + [raw[0]]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_records(path)


def test_target_packs_and_rotation(tokenizer):
    row = {"id": "a", "document": ("judgment reasoning dismissed " * 1100),
           "summary": "Facts judgment reasoning dismissed. " * 150}
    inputs = PassageInputs(tokenizer)
    assert len(inputs.prompts(row["document"])) == 4
    v3 = training_group(row, inputs, "v3")
    v2 = training_group(row, inputs, "v2")
    v1 = training_group(row, SectionInputs(tokenizer), "original")
    assert len(v3) > 1
    assert len(v2) == len(v1) == 1
    assert v1[0]["tgt"][-1] == EOS and len(v1[0]["tgt"]) == 240
    assert all(ex["src"][0] == BOS and ex["tgt"][-1] == EOS for ex in v3)
    assert all(len(ex["tgt"]) <= 320 and len(ex["src"]) + len(ex["tgt"]) <= 1280 for ex in v3)
    groups = [list(range(4)), list(range(2))]
    plan = schedule(groups, configuration("v3", "E05"), 42)
    assert len(plan) == len(set(plan)) == 6
    plan_v2 = schedule(groups, configuration("v2", "E05"), 42)
    assert len(plan_v2) == 6 and (0, 3) not in plan_v2


@pytest.mark.parametrize("version,name", [
    ("original", "E01"), ("v3", "E01"), ("v3", "E02"), ("original", "E03"),
    ("v3", "E03"), ("v3", "E04"), ("v3", "E05"), ("v3", "E06"), ("v3", "E07_gold"),
])
def test_shapes_gradients_and_strict_checkpoint(version, name, tmp_path):
    run = configuration(version, name)
    model = configure_model(tiny(run.architecture), run)
    logits = target_logits(model, run.architecture, [1, 4, 5], [6, 8, 2], "cpu")
    assert logits.shape == (3, 10) and torch.isfinite(logits).all()
    logits.float().square().mean().backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = configure_model(tiny(run.architecture), run)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_distillation_and_decode_constraints(tokenizer):
    student = torch.randn(3, 10, requires_grad=True)
    teacher = student.detach().clone().requires_grad_(True)
    expected = .8 * nn.functional.cross_entropy(student, torch.tensor([1, 4, 2]))
    loss = distillation_loss(student, teacher, [1, 4, 2])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert teacher.grad is None and torch.isfinite(student.grad).all()
    logits = torch.tensor([[[4., 4., 3., 4., 2., 1., 0., 0., 0., 0.]]])
    assert next_token(logits, [], 0, 32, 1.1) == 4
    assert next_token(logits, [4, 4, 4], 3, 32, 1.1) == 5
    assert next_token(logits, [], 32, 32, 1.1) == EOS
    model = tiny("efficient_gqa")
    args = (model, "efficient_gqa", [1, 4], tokenizer, "cpu")
    assert generate_passage(*args, max_new_tokens=5) == generate_passage(*args, max_new_tokens=5)


def test_aggregation_and_submission(tmp_path):
    assert dedupe(["Evidence supports relief.", "Evidence supports relief."]) == "Evidence supports relief."
    assert evidence_filter("Invented amount 2000.", "The award was 1300.") == "Invented amount 2000."
    rows = [{"id": "a"}, {"id": "b"}]
    path = tmp_path / "submission.zip"
    write_submission(path, rows, {"a": "Summary A.", "b": "Summary B."})
    assert [row["id"] for row in read_submission(path, rows)] == ["a", "b"]
    with pytest.raises(ValueError):
        write_submission(path, rows, {"a": " "})
    with ZipFile(path, "a") as archive:
        archive.writestr("extra.json", "{}")
    with pytest.raises(ValueError):
        read_submission(path, rows)


def test_training_resumes_without_extra_updates(tmp_path, monkeypatch, tokenizer):
    monkeypatch.setattr("phase6.training.make_model", lambda run, foundation: tiny(run.architecture))
    run = replace(configuration("original", "E02"), steps=2, validate_every=1, save_every=1)
    pair = {"id": "a", "key": "a", "src": [1, 4, 5], "tgt": [6, 8, 2]}
    args = (run, [[pair]], [pair], tokenizer, tmp_path, "cpu", "fp32", 42, "test")
    path = train(*args, foundation="synthetic")
    state = torch.load(tmp_path / "resume.pt", weights_only=True)
    assert path.is_file() and state["step"] == 2
    assert train(*args, foundation="synthetic") == path
    assert [row["step"] for row in json.loads((tmp_path / "history.json").read_text())] == [1, 2]
    with pytest.raises(ValueError, match="changed"):
        train(*args[:-1], "different-inputs", foundation="synthetic")
