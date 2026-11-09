"""Research configurations for the three Phase 6 training protocols."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from phase1.models import build_model
from .adaptation import LoRALinear, add_low_rank

VERSIONS = {
    "original": None,
    "v2": "v2_passage_align_20261101",
    "v3": "v3_allpassages_kl_20261101_v1",
}
METHODS = {
    "E01": ("encoder_decoder", "lora"),
    "E02": ("efficient_gqa", "lorma"),
    "E03": ("prefix_lm", "houlsby"),
    "E04": ("decoder_only", "distillation"),
    "E05": ("efficient_gqa", "lora"),
    "E06": ("encoder_decoder", "full"),
    "E07_gold": ("decoder_only", "student_gold"),
}


@dataclass(frozen=True)
class Run:
    version: str
    name: str
    architecture: str
    method: str
    lr: float
    epochs: int
    steps: int | None
    validate_every: int
    save_every: int
    nll_pairs: int
    validation_docs: int


def configuration(version, name):
    architecture, method = METHODS[name]
    student = method in ("distillation", "student_gold")
    if version == "original" and name != "E07_gold":
        rate = 4e-5 if student else 2e-5 if method == "full" else 2e-4
        steps = {"E05": 650, "E06": 500}.get(name, 800)
        return Run(version, name, architecture, method, rate, 0, steps, 100, 50, 24, 32)
    if version == "v2":
        rate = 4e-5 if student else 2e-5 if method == "full" else 1.5e-4
        return Run(version, name, architecture, method, rate, 3, None, 400, 200, 48, 64)
    if version == "v3":
        rate = 2e-5 if student else 1e-5 if method == "full" else 8e-5
        return Run(version, name, architecture, method, rate, 1, None, 250, 100, 64, 96)
    raise ValueError(f"Unsupported run {version}/{name}")


class LoRMALinear(nn.Module):
    """The notebook's multiplicative update on the frozen layer output."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Linear(base.out_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        z = self.base(x)
        return z + self.scale * self.B(self.A(z))


class HoulsbyAdapter(nn.Module):
    def __init__(self, hidden, original=False):
        super().__init__()
        self.norm = nn.LayerNorm(hidden) if original else nn.Identity()
        self.down = nn.Linear(hidden, 64)
        self.up = nn.Linear(64, hidden)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(F.relu(self.down(self.norm(x))))


class OriginalAdapterBlock(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.adapter = HoulsbyAdapter(base.norm1.normalized_shape[0], original=True)

    def forward(self, *args, **kwargs):
        hidden, cache = self.base(*args, **kwargs)
        return hidden + self.adapter(hidden), cache


class HoulsbyPrefixBlock(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        hidden = base.norm1.normalized_shape[0]
        self.attention_adapter = HoulsbyAdapter(hidden)
        self.ffn_adapter = HoulsbyAdapter(hidden)

    def forward(self, x, causal, attention_mask=None, past_kv=None, use_cache=False):
        y, cache = self.base.attention(self.base.norm1(x), causal=causal,
                                       attention_mask=attention_mask, past_kv=past_kv,
                                       use_cache=use_cache)
        x = x + self.base.dropout(y)
        x = x + self.attention_adapter(x)
        x = x + self.base.mlp(self.base.norm2(x))
        return x + self.ffn_adapter(x), cache


def load_state(path):
    """Read a trusted historical model-only checkpoint."""
    return torch.load(path, map_location="cpu", weights_only=True)


def configure_model(model, run):
    original = run.version == "original"
    if run.method in ("lora", "lorma"):
        return add_low_rank(model, LoRALinear if run.method == "lora" else LoRMALinear, original)
    if run.method == "houlsby":
        model.requires_grad_(False)
        wrap = OriginalAdapterBlock if original else HoulsbyPrefixBlock
        model.blocks = nn.ModuleList(wrap(block) for block in model.blocks)
        if not original:
            for layer in model.modules():
                if isinstance(layer, nn.LayerNorm):
                    layer.requires_grad_(True)
    if run.method in ("distillation", "student_gold"):
        keep = np.linspace(0, len(model.blocks) - 1, 8).round().astype(int)
        model.blocks = nn.ModuleList(model.blocks[int(i)] for i in keep)
    return model


def make_model(run, foundation=None):
    model = build_model(run.architecture)
    if foundation is not None:
        # Historical foundation archives include NumPy/RNG state alongside model tensors.
        state = torch.load(foundation, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"], strict=True)
    return configure_model(model, run)


def restore_model(run, checkpoint, device):
    state = load_state(checkpoint)
    if state.get("exp_id") != run.name or state.get("version") != VERSIONS[run.version]:
        raise ValueError(f"Expected {run.version}/{run.name} checkpoint")
    model = make_model(run)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()
