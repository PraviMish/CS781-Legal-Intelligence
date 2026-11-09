"""Historical Phase 5 adaptations of the four Phase 1 architectures."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from phase1.models import build_model
from .classifier import StatuteClassifier


@dataclass(frozen=True)
class Run:
    name: str
    arch: str
    method: str
    loss: str
    lr: float
    epochs: int
    rank: int = 8
    student: bool = False
    teacher: str | None = None

    @property
    def eval_batch_size(self):
        return 1 if self.arch in ("prefix_lm", "encoder_decoder") else 2


RUNS = {
    run.name[:3]: run for run in (
        Run("E01_GQA_LoRMA", "efficient_gqa", "lorma", "weighted_bce", 4e-4, 4),
        Run("E02_GQA_FullFT", "efficient_gqa", "full", "asl", 2e-5, 3),
        Run("E03_Decoder_Partial", "decoder_only", "partial", "weighted_bce", 8e-5, 4),
        Run("E04_Prefix_Adapters", "prefix_lm", "adapter", "weighted_bce", 4e-4, 4),
        Run("E05_EncDec_LoRA", "encoder_decoder", "lora", "asl", 4e-4, 4),
        Run("E06_GQA_Distilled", "efficient_gqa", "full", "weighted_bce", 4e-5, 4,
            student=True, teacher="E01_GQA_LoRMA"),
    )
}
STUDENT_LAYERS = (0, 3, 5, 8, 10, 13)


class LowRankProjection(nn.Module):
    def __init__(self, base, rank, on_output):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = 16 / rank
        self.on_output = on_output
        width = base.out_features if on_output else base.in_features
        self.A = nn.Linear(width, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        y = self.base(x)
        return y + self.scale * self.B(self.A(y if self.on_output else x))


class BottleneckProjection(nn.Module):
    def __init__(self, base, width):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(base.out_features, width)
        self.up = nn.Linear(width, base.out_features)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        y = self.base(x)
        return y + self.up(F.gelu(self.down(y)))


def adapt(model, method, rank):
    if method == "full":
        return
    backbone = model.backbone.requires_grad_(False)
    if method == "partial":
        for block in backbone.blocks[-4:]:
            block.requires_grad_(True)
        backbone.final_norm.requires_grad_(True)
        return
    projections = {"q_proj", "k_proj", "v_proj", "out_proj", "qkv"}
    selected = [(name, layer) for name, layer in backbone.named_modules()
                if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in projections]
    if not selected or method not in ("lorma", "lora", "adapter"):
        raise ValueError(f"No supported projections for {method}")
    for name, layer in selected:
        parent, _, attribute = name.rpartition(".")
        wrapper = (BottleneckProjection(layer, max(32, rank * 4)) if method == "adapter"
                   else LowRankProjection(layer, rank, on_output=method == "lorma"))
        setattr(backbone.get_submodule(parent), attribute, wrapper)


def make_model(run, foundation=None):
    backbone = build_model(run.arch)
    if foundation is not None:
        saved = torch.load(foundation, map_location="cpu", weights_only=True, mmap=True)
        if saved.get("model_name") != run.arch:
            raise ValueError("Phase 1 checkpoint architecture does not match the experiment")
        backbone.load_state_dict(saved["model_state"], strict=True)
    if run.student:
        backbone.blocks = nn.ModuleList(backbone.blocks[index] for index in STUDENT_LAYERS)
    model = StatuteClassifier(backbone, run.arch)
    adapt(model, run.method, run.rank)
    return model


def restore_model(run, checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config = saved["config"]
    for key in ("arch", "method", "student", "rank"):
        actual = config.get(key, {"student": False, "rank": 8}.get(key))
        if actual != getattr(run, key):
            raise ValueError(f"Checkpoint does not match {run.name}: {key}")
    model = make_model(run)
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval()
