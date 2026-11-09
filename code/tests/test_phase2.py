"""Regression checks for Phase 2 offsets, masking, loss, voting and checkpoint keys."""
from dataclasses import replace

import pytest
import torch

from phase1.models import EncoderDecoderTransformer, PrefixLMTransformer
from phase2.configs import FULL70_MEMBERS, GQA_FULL, RUNS
from phase2.data import IGNORE, TAG_ID, build_windows, decode_bio, normalize_gold, spans_to_bio, trim_offsets
from phase2.ensemble import ensemble_documents
from phase2.metrics import strict_metrics
from phase2.models import LegalNER, LoRMAPlusLinear, ner_weights, restore_ner
from phase2.runtime import predict_documents
from phase2.submission import load_predictions, validate_predictions, write_submission
from phase2.train import supervised_loss, train


def test_bio_alignment_preserves_offsets_and_ignores_unrepresentable_entities():
    text = "A Supreme  Court."
    offsets = trim_offsets(text, [(0, 1), (1, 9), (9, 10), (10, 16), (16, 17)])
    gold = normalize_gold({"x": text}, {"x": [[2, 16, "COURT", "Supreme  Court"]]})
    tags, represented = spans_to_bio(offsets, gold["x"])
    assert represented == 1
    assert tags == [0, TAG_ID["B-COURT"], IGNORE, TAG_ID["I-COURT"], 0]
    assert decode_bio([max(0, tag) for tag in tags], offsets) == gold["x"]
    tags, represented = spans_to_bio(offsets, [[3, 16, "COURT"]])
    assert represented == 0
    assert tags == [0, IGNORE, IGNORE, IGNORE, 0]


class EchoTags(torch.nn.Module):
    pad_id = 0

    def forward(self, inputs):
        return torch.nn.functional.one_hot(inputs, num_classes=25).float()


def test_owned_windows_decode_entity_across_seam():
    tags = [0] * 600
    tags[255:258] = [TAG_ID["B-DATE"], TAG_ID["I-DATE"], TAG_ID["I-DATE"]]
    docs = {"doc": {"ids": tags, "labels": tags, "offsets": [(i, i + 1) for i in range(600)]}}
    windows = build_windows(docs, docs)
    assert [len(item["input_ids"]) for item in windows] == [384, 472, 216]
    assert [tag for item in windows for tag in item["labels"] if tag != IGNORE] == tags
    assert predict_documents(EchoTags(), docs, docs, 2, torch.device("cpu")) == {
        "doc": [[255, 258, "DATE"]]
    }


@pytest.mark.parametrize("method", ["lora", "lorma"])
def test_adapter_freeze_padding_and_historical_keys(method):
    if method == "lora":
        backbone = EncoderDecoderTransformer(
            vocab_size=24, encoder_max_len=16, decoder_max_len=16, d_model=8,
            n_encoder_layers=1, n_decoder_layers=1, n_heads=2, d_ff=16, dropout=0.1
        )
        architecture = "encoder_decoder"
    else:
        backbone = PrefixLMTransformer(
            vocab_size=24, max_seq_len=16, d_model=8, n_layers=1, n_heads=2, d_ff=16, dropout=0.1
        )
        architecture = "prefix_lm"
    model = LegalNER(backbone, architecture, method, pad_id=0)
    model.train()
    assert model.backbone.training == (method == "lora")
    model.eval()
    logits = model(torch.tensor([[1, 2, 3]]))
    padded = model(torch.tensor([[1, 2, 3, 0, 0]]))
    torch.testing.assert_close(logits, padded[:, :3])
    weights = ner_weights(model)
    assert len(weights["trainable_state"]) == 6
    restore_ner(model, weights)
    model(torch.tensor([[1, 2, 3]])).sum().backward()
    assert all(param.grad is None for param in backbone.parameters() if not param.requires_grad)
    weights["trainable_state"].pop("classifier.bias")
    with pytest.raises(ValueError, match="tensor names"):
        restore_ner(model, weights)


def test_lorma_transforms_frozen_output_before_bias():
    layer = LoRMAPlusLinear(torch.nn.Linear(3, 4), rank=2, alpha=6)
    x = torch.randn(2, 3)
    torch.testing.assert_close(layer(x), layer.base(x))
    with torch.no_grad():
        layer.lorma_B.fill_(0.2)
    z = x @ layer.base.weight.T
    expected = z + 3 * z @ layer.lorma_A.T @ layer.lorma_B.T + layer.base.bias
    torch.testing.assert_close(layer(x), expected)


def test_kd_uses_only_owned_labels_and_supervised_weight():
    logits = torch.randn(1, 3, 25)
    labels = torch.tensor([[1, IGNORE, IGNORE]])
    teacher = torch.randn_like(logits)
    expected = supervised_loss(logits, labels, teacher=teacher)
    teacher[:, 1:] = -500
    torch.testing.assert_close(supervised_loss(logits, labels, teacher=teacher), expected)
    torch.testing.assert_close(
        supervised_loss(logits, labels, teacher=logits), 0.7 * supervised_loss(logits, labels)
    )


def test_vote_deduplicates_per_model_and_prefers_gqa_on_equal_support(tmp_path):
    members = {
        "gqa": {"doc": [[2, 5, "APP"], [9, 10, "WIT"], [9, 10, "WIT"]]},
        "decoder": {"doc": [[2, 5, "APP"], [1, 6, "RESP"]]},
        "student": {"doc": [[1, 6, "RESP"]]},
    }
    result = ensemble_documents(members)
    assert result == {"doc": [[2, 5, "APP"]]}
    assert ensemble_documents(members, "unanimous") == {"doc": []}
    assert ensemble_documents(members, "teacher_plus_agreement")["doc"][-1] == [9, 10, "WIT"]
    summary, classes = strict_metrics({"doc": [[1, 6, "APP"]]}, result)
    assert summary["macro_f1"] == 0
    assert classes["APP"]["FP"] == classes["APP"]["FN"] == 1
    path = tmp_path / "submission.zip"
    texts = {"doc": "0123456789"}
    write_submission(path, result, texts, nonoverlapping=True)
    assert load_predictions(path) == result
    with pytest.raises(ValueError, match="nonoverlapping"):
        validate_predictions({"doc": [[2, 5, "APP"], [3, 6, "APP"]]}, texts,
                             nonoverlapping=True)


def test_train_resume_and_full_data_checkpoint_selection(tmp_path, monkeypatch):
    import importlib
    trainer = importlib.import_module("phase2.train")

    class TinyNER(torch.nn.Module):
        method = "full"
        pad_id = 0

        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Embedding(5, 8)
            self.classifier = torch.nn.Linear(8, 25)

        def forward(self, inputs):
            return self.classifier(self.backbone(inputs))

    monkeypatch.setattr(trainer, "build_ner", lambda *args, **kwargs: TinyNER())
    docs = {key: {"ids": [1, 2], "labels": [0, 1], "offsets": [(0, 1), (1, 2)]}
            for key in ("train", "val")}
    gold = {key: [[1, 2, "APP"]] for key in docs}
    device = torch.device("cpu")
    holdout = replace(RUNS[GQA_FULL], epochs=2, batch=1, accumulation=1, save_every=1)
    split = {"train": ["train"], "val": ["val"]}
    assert train(GQA_FULL, holdout, docs, gold, split, 0, 42, tmp_path / "val", device).is_file()
    assert train(GQA_FULL, holdout, docs, gold, split, 0, 42, tmp_path / "val", device).name == "best.pt"
    with pytest.raises(ValueError, match="Run inputs changed"):
        train(GQA_FULL, holdout, docs, gold, split, 0, 43, tmp_path / "val", device)
    full_name = FULL70_MEMBERS[0]
    full = replace(RUNS[full_name], epochs=1, batch=1, accumulation=1)
    split = {"train": ["train", "val"], "val": []}
    selected = train(full_name, full, docs, gold, split, 0, 42, tmp_path / "full", device)
    assert selected.is_file() and selected.name == "final.pt"
