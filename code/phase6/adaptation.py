"""Low-rank layers with the original checkpoint's base/A/B parameter names."""

import math

from torch import nn

from phase1.models import RMSNorm


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


def add_low_rank(model, layer_type=LoRALinear, original=False):
    model.requires_grad_(False)
    targets = []
    for name, layer in model.named_modules():
        if not isinstance(layer, nn.Linear) or not name.startswith(
            ("blocks.", "encoder_blocks.", "decoder_blocks.")
        ):
            continue
        if original:
            selected = name.endswith(("qkv", "q_proj", "k_proj", "v_proj", "out_proj"))
        else:
            selected = any(part in name for part in
                           (".attention.", ".self_attention.", ".cross_attention.", ".mlp."))
        if selected:
            targets.append((name, layer))
    if not targets:
        raise ValueError("No eligible projections for low-rank adaptation")
    rank = 8 if original else 16
    for name, layer in targets:
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, layer_type(layer, rank, 2 * rank))
    if not original:
        for layer in model.modules():
            if isinstance(layer, (nn.LayerNorm, RMSNorm)):
                layer.requires_grad_(True)
    return model
