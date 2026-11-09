"""Dispatch historical run models and build frozen E02 features."""

from dataclasses import dataclass

import numpy as np
import torch

from .inference import amp, backbone, e01_model, load_checkpoint
from .variants import (
    E02HierarchicalHead,
    E03GQALoRMAClassifier,
    E04DecoderClassifier,
    E05PrefixClassifier,
    E06EncoderDecoderClassifier,
    E07GQAClassifier,
    E08SmallTransformer,
)

ARCHITECTURES = {
    "E03": "efficient_gqa",
    "E04": "decoder_only",
    "E05": "prefix_lm",
    "E06": "encoder_decoder",
    "E07": "efficient_gqa",
}
STATE_KEYS = {"E02": "head", "E03": "trainable_state", "E07": "adapters", "E08": "student"}
BEST_FILES = {
    "E02": "best_head.pt",
    "E03": "best_adapters.pt",
    "E07": "best_adapters.pt",
    "E08": "best_student.pt",
}
ADAPTER_RUNS = ("E03", "E07")


def create_model(run, *, checkpoint=None, foundation=None, bos_id=None):
    if run == "E01":
        return e01_model(checkpoint, foundation)
    if run == "E02":
        model = E02HierarchicalHead()
    elif run == "E08":
        model = E08SmallTransformer()
    else:
        if foundation is None and (checkpoint is None or run in ADAPTER_RUNS):
            raise ValueError(f"{run} requires its original Phase 1 foundation")
        architecture = ARCHITECTURES[run]
        tokens = 700047360 if architecture == "efficient_gqa" else 100024320
        base = backbone(architecture, foundation, tokens)
        if run == "E06":
            if bos_id is None:
                raise ValueError("E06 requires the archived tokenizer BOS ID")
            model = E06EncoderDecoderClassifier(base, bos_id)
        else:
            classifier = {
                "E03": E03GQALoRMAClassifier,
                "E04": E04DecoderClassifier,
                "E05": E05PrefixClassifier,
                "E07": E07GQAClassifier,
            }[run]
            model = classifier(base)
    if checkpoint is not None:
        restore_weights(model, run, load_checkpoint(checkpoint)[STATE_KEYS.get(run, "model")])
    return model


def weight_keys(model, run):
    if run == "E03":
        return {name for name, p in model.named_parameters() if p.requires_grad}
    return {
        name
        for name in model.state_dict()
        if name.startswith("classifier.")
        or name.rsplit(".", 1)[-1] in {"lora_A", "lora_B", "lora_E", "rank_mask", "score_ema"}
    }


def saved_weights(model, run):
    weights = model.state_dict()
    if run in ADAPTER_RUNS:
        return {key: weights[key] for key in sorted(weight_keys(model, run))}
    return weights


def restore_weights(model, run, weights):
    if run in ADAPTER_RUNS:
        expected = weight_keys(model, run)
        if set(weights) != expected:
            raise ValueError(f"{run} checkpoint has unexpected or missing adapter keys")
        state = model.state_dict()
        state.update(weights)
        weights = state
    model.load_state_dict(weights, strict=True)


@dataclass
class Features:
    split: object
    features: np.ndarray
    logits: np.ndarray

    @property
    def ids(self):
        return self.split.ids

    @property
    def labels(self):
        return self.split.labels

    def batch(self, indices, device):
        return (
            torch.from_numpy(self.features[indices]).to(device),
            torch.from_numpy(self.logits[indices]).to(device),
            torch.from_numpy(self.split.lengths[indices] > 0).to(device),
        )


@torch.inference_mode()
def extract_features(teacher, split, device, precision):
    """Match the notebook's 12-case and 12-window inference batches."""
    teacher.eval()
    if split.tokens.shape[1] != 5:
        raise ValueError("E02 requires the hierarchical five-slot cache")
    features = np.zeros((len(split.ids), 5, 768), dtype=np.float16)
    logits = np.zeros((len(split.ids), 5, 2), dtype=np.float16)
    for start in range(0, len(split.ids), 12):
        end = min(start + 12, len(split.ids))
        rows, slots = np.nonzero(split.lengths[start:end] > 0)
        for offset in range(0, len(rows), 12):
            r = start + rows[offset : offset + 12]
            s = slots[offset : offset + 12]
            lengths = split.lengths[r, s].astype(np.int64)
            ids = torch.from_numpy(split.tokens[r, s, : lengths.max()].astype(np.int64)).to(device)
            with amp(device, precision):
                hidden, _ = teacher.backbone.hidden_states(ids)
                pooled = hidden[torch.arange(len(r), device=device), lengths - 1]
                chunk_logits = teacher.classifier(pooled)
            features[r, s] = pooled.float().cpu().numpy()
            logits[r, s] = chunk_logits.float().cpu().numpy()
    if not np.isfinite(features).all() or not np.isfinite(logits).all():
        raise FloatingPointError("Non-finite cached teacher features")
    return Features(split, features, logits)
