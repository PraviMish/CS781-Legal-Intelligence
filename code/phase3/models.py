"""Exact Phase-3 classifier heads and notebook adaptation parameterizations.

Imports the shared Phase-1 architectures without repackaging or changing model weights.
Run with the repository's code/ directory on PYTHONPATH.
"""
from __future__ import annotations
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from phase1.models import build_model

STUDENT_LAYER_INDICES = (0, 2, 4, 6, 8, 11)


class LoRALinear(nn.Module):
    def __init__(self, original, rank=8, alpha=16, dropout=.05):
        super().__init__()
        if not isinstance(original, nn.Linear):
            raise TypeError("LoRA wraps nn.Linear")
        self.base = original
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, original.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.requires_grad_(False)
    def forward(self, x):
        return self.base(x) + self.scaling * F.linear(
            F.linear(self.dropout(x), self.lora_A), self.lora_B)


class LoRMAPlusLinear(nn.Module):
    """Output-side identity initialized multiplicative (I + BA) W transformation.

    In this project all wrapped GQA projections are bias-free.
    """
    def __init__(self, original, rank=8, alpha=16):
        super().__init__()
        if not isinstance(original, nn.Linear):
            raise TypeError("LoRMA+ wraps nn.Linear")
        self.base = original
        self.scaling = alpha / rank
        width = original.out_features
        self.lorma_A = nn.Parameter(torch.empty(rank, width))
        self.lorma_B = nn.Parameter(torch.zeros(width, rank))
        nn.init.kaiming_uniform_(self.lorma_A, a=math.sqrt(5))
        self.base.requires_grad_(False)
    def forward(self, x):
        y = self.base(x)
        return y + self.scaling * F.linear(F.linear(y, self.lorma_A), self.lorma_B)


class IA3Linear(nn.Module):
    def __init__(self, original, scale_input=False):
        super().__init__()
        self.base = original
        self.scale_input = scale_input
        width = original.in_features if scale_input else original.out_features
        self.ia3_scale = nn.Parameter(torch.ones(width))
        self.base.requires_grad_(False)
    def forward(self, x):
        return self.base(x * self.ia3_scale) if self.scale_input else self.base(x) * self.ia3_scale


class PrefixGQAAttention(nn.Module):
    """Train virtual K/V tokens per original GQA attention layer (prefix length 16)."""
    def __init__(self, original, prefix_length=16):
        super().__init__()
        self.base = original
        self.prefix_length = prefix_length
        heads, d = original.n_kv_heads, original.head_dim
        self.prefix_key = nn.Parameter(torch.randn(1, heads, prefix_length, d) * .02)
        self.prefix_value = nn.Parameter(torch.randn(1, heads, prefix_length, d) * .02)
        self.base.requires_grad_(False)

    def forward(self, x, past_kv=None, use_cache=False):
        if past_kv is not None or use_cache:
            raise NotImplementedError("Original prefix classifier only implements noncached attention")
        B, T, _ = x.shape
        attn = self.base
        hq, hk, d, P = attn.n_query_heads, attn.n_kv_heads, attn.head_dim, self.prefix_length
        q = attn.q_proj(x).view(B, T, hq, d).transpose(1, 2)
        k = attn.k_proj(x).view(B, T, hk, d).transpose(1, 2)
        v = attn.v_proj(x).view(B, T, hk, d).transpose(1, 2)
        q, k = attn.rope.apply_rotary(q, k, 0)
        k = torch.cat((self.prefix_key.expand(B, -1, -1, -1).to(k.dtype), k), dim=2)
        v = torch.cat((self.prefix_value.expand(B, -1, -1, -1).to(v.dtype), v), dim=2)
        mask = torch.cat((torch.ones(T, P, dtype=torch.bool, device=x.device),
                          torch.ones(T, T, dtype=torch.bool, device=x.device).tril()), dim=1)
        try:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0., enable_gqa=True)
        except TypeError:
            copies = hq // hk
            out = F.scaled_dot_product_attention(
                q, k.repeat_interleave(copies, dim=1), v.repeat_interleave(copies, dim=1),
                attn_mask=mask, dropout_p=0.)
        out = out.transpose(1, 2).contiguous().view(B, T, attn.d_model)
        return attn.out_proj(out), None


class LegalRRClassifier(nn.Module):
    """Classify from final nonpadding EOS; preserve notebook's attention semantics."""
    def __init__(self, backbone, architecture, hidden_size, num_labels=13):
        super().__init__()
        self.backbone = backbone
        self.architecture = architecture
        self.head = nn.Sequential(nn.Dropout(.15), nn.Linear(hidden_size, num_labels))

    def sentence_representations(self, input_ids, lengths):
        B, T = input_ids.shape
        net, device = self.backbone, input_ids.device
        if self.architecture in ("decoder_only", "efficient_gqa"):
            # Causality protects the final real token from right-side padding.
            hidden, _ = net.hidden_states(input_ids)
        elif self.architecture == "prefix_lm":
            positions = torch.arange(T, device=device)
            hidden = net.dropout(net.token_embedding(input_ids) + net.position_embedding(positions)[None])
            valid = torch.arange(T, device=device)[None] < lengths[:, None]
            attention_mask = valid[:, None, None, :]
            for block in net.blocks:
                hidden, _ = block(hidden, causal=False, attention_mask=attention_mask)
            hidden = net.final_norm(hidden)
        elif self.architecture == "encoder_decoder":
            positions = torch.arange(T, device=device)
            hidden = net.dropout(net.token_embedding(input_ids) + net.encoder_position(positions)[None])
            valid = torch.arange(T, device=device)[None] < lengths[:, None]
            attention_mask = valid[:, None, None, :]
            for block in net.encoder_blocks:
                out, _ = block.attention(block.norm1(hidden), causal=False, attention_mask=attention_mask)
                hidden = hidden + block.dropout(out)
                hidden = hidden + block.mlp(block.norm2(hidden))
            hidden = net.encoder_norm(hidden)
        else:
            raise ValueError(self.architecture)
        return hidden[torch.arange(B, device=device), lengths - 1, :]

    def forward(self, input_ids, lengths):
        return self.head(self.sentence_representations(input_ids, lengths))


def inject_lora(backbone, architecture, rank=8, alpha=16):
    targets = []
    for name, module in list(backbone.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if architecture == "encoder_decoder":
            if not name.startswith("encoder_blocks."):
                continue
        elif not name.startswith("blocks."):
            continue
        if name.endswith((".qkv", ".q_proj", ".k_proj", ".v_proj", ".out_proj")):
            parent, attr = name.rsplit(".", 1)
            setattr(backbone.get_submodule(parent), attr, LoRALinear(module, rank, alpha))
            targets.append(name)
    if not targets:
        raise ValueError("No matching LoRA attention projections")
    return targets


def inject_gqa_method(backbone, method):
    if method == "full":
        backbone.requires_grad_(True)
        return ["all_pretrained_weights"]
    if method in ("lora", "lorma_plus"):
        targets = []
        for name, module in list(backbone.named_modules()):
            if name.startswith("blocks.") and isinstance(module, nn.Linear) and name.endswith(
                (".q_proj", ".k_proj", ".v_proj", ".out_proj")):
                parent, attr = name.rsplit(".", 1)
                cls = LoRALinear if method == "lora" else LoRMAPlusLinear
                setattr(backbone.get_submodule(parent), attr, cls(module, rank=8, alpha=16))
                targets.append(name)
        return targets
    if method == "ia3":
        targets = []
        for i, block in enumerate(backbone.blocks):
            for projection in ("k_proj", "v_proj"):
                src = getattr(block.attention, projection)
                setattr(block.attention, projection, IA3Linear(src, scale_input=False))
                targets.append(f"blocks.{i}.attention.{projection}")
            block.mlp.down = IA3Linear(block.mlp.down, scale_input=True)
            targets.append(f"blocks.{i}.mlp.down")
        return targets
    if method == "prefix":
        for i, block in enumerate(backbone.blocks):
            block.attention = PrefixGQAAttention(block.attention, prefix_length=16)
        return [f"blocks.{i}.attention" for i in range(len(backbone.blocks))]
    raise ValueError(method)


def load_foundation(backbone, path):
    """Load original Phase-1 checkpoint; reject incompatible tensor names or sizes."""
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if "model_state" not in state:
        raise ValueError("Expected Phase-1 checkpoint containing model_state")
    backbone.load_state_dict(state["model_state"], strict=True)
    return state.get("tokens_seen")


def make_classifier(architecture, method, foundation, *, device="cpu", student=False):
    """Rebuild notebook model and load its original Phase-1 starting point.

    Run with a frozen original foundation and optional small adapter/head checkpoint.
    A missing foundation is an error; no silent random initialization occurs.
    """
    architecture = "decoder_only" if student else architecture
    backbone = build_model(architecture)
    load_foundation(backbone, foundation)
    if student:
        blocks = list(backbone.blocks)
        backbone.blocks = nn.ModuleList([blocks[i] for i in STUDENT_LAYER_INDICES])
        backbone.requires_grad_(True)
    else:
        backbone.requires_grad_(False)
        if method == "lora":
            if architecture == "efficient_gqa":
                inject_gqa_method(backbone, method)
            else:
                inject_lora(backbone, architecture)
        else:
            if architecture != "efficient_gqa":
                raise ValueError("Notebook non-LoRA adaptation used Efficient GQA only")
            inject_gqa_method(backbone, method)
    hidden_size = backbone.token_embedding.embedding_dim
    model = LegalRRClassifier(backbone, architecture, hidden_size, 13).to(device)
    return model


def trainable_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def restore_trainable(model, state):
    """Strictly check adapter/head names; never load mismatched initialization."""
    names = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if names.keys() != state.keys():
        raise ValueError(f"Trainable tensor keys differ: {names.keys() ^ state.keys()}")
    with torch.no_grad():
        for name, target in names.items():
            if target.shape != state[name].shape:
                raise ValueError(f"Tensor shape mismatch for {name}")
            target.copy_(state[name].to(device=target.device, dtype=target.dtype))
