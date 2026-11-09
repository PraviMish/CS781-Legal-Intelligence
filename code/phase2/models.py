"""BIO heads and the original Phase 2 adapter parameterizations."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from phase1.models import EfficientGQATransformer, build_model
from .configs import DECODER, STUDENT_LAYERS
from .data import BIO


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.empty(base.out_features, rank))
        self.reset_adapter()

    def reset_adapter(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, inputs):
        projected = F.linear(self.lora_dropout(inputs), self.lora_A)
        return self.base(inputs) + self.scale * F.linear(projected, self.lora_B)


class LoRMAPlusLinear(nn.Module):
    """The notebook's output-side residual transform of the frozen linear map."""
    def __init__(self, base, rank=16, alpha=32):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        width = base.out_features
        self.lorma_A = nn.Parameter(torch.empty(rank, width))
        self.lorma_B = nn.Parameter(torch.zeros(width, rank))
        nn.init.kaiming_uniform_(self.lorma_A, a=math.sqrt(5))

    def forward(self, inputs):
        base_output = F.linear(inputs, self.base.weight)
        update = F.linear(F.linear(base_output, self.lorma_A), self.lorma_B)
        output = base_output + self.scale * update
        if self.base.bias is not None:
            output = output + self.base.bias
        return output


class LegalNER(nn.Module):
    def __init__(self, backbone, architecture, method, pad_id):
        super().__init__()
        self.backbone = backbone
        self.architecture = architecture
        self.method = method
        self.pad_id = pad_id
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(backbone.token_embedding.embedding_dim, len(BIO))

        if method in {"probe", "lora", "lorma"}:
            self.backbone.requires_grad_(False)
        if method in {"lora", "lorma"}:
            blocks = backbone.encoder_blocks if method == "lora" else backbone.blocks
            adapter = LoRALinear if method == "lora" else LoRMAPlusLinear
            for block in blocks:
                block.attention.qkv = adapter(block.attention.qkv)
                block.attention.out_proj = adapter(block.attention.out_proj)

    def train(self, mode=True):
        super().train(mode)
        if self.method in {"probe", "lorma"}:
            self.backbone.eval()
        return self

    def forward(self, input_ids):
        if self.method == "probe":
            with torch.no_grad():
                hidden, _ = self.backbone.hidden_states(input_ids)
        elif self.method == "lora":
            hidden = self._encoder(input_ids)
        else:
            hidden = self._transformer(input_ids)
        return self.classifier(self.dropout(hidden))

    def _encoder(self, input_ids):
        backbone = self.backbone
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = backbone.token_embedding(input_ids) + backbone.encoder_position(positions)[None]
        hidden = backbone.dropout(hidden)
        key_mask = (input_ids != self.pad_id)[:, None, None, :]
        for block in backbone.encoder_blocks:
            attention, _ = block.attention(
                block.norm1(hidden), causal=False, attention_mask=key_mask
            )
            hidden = hidden + block.dropout(attention)
            hidden = hidden + block.mlp(block.norm2(hidden))
        return backbone.encoder_norm(hidden)

    def _transformer(self, input_ids):
        backbone = self.backbone
        hidden = backbone.token_embedding(input_ids)
        if self.architecture == DECODER or self.method == "lorma":
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            hidden = backbone.dropout(hidden + backbone.position_embedding(positions)[None])
        block_options = {"use_cache": False}
        if self.architecture == DECODER:
            block_options["causal"] = True
        elif self.method == "lorma":
            block_options["causal"] = False
            block_options["attention_mask"] = (input_ids != self.pad_id)[:, None, None, :]

        for block in backbone.blocks:
            def block_hidden(inputs, current=block):
                output, _ = current(inputs, **block_options)
                return output

            if self.training and self.method != "lorma":
                hidden = checkpoint(block_hidden, hidden, use_reentrant=False)
            else:
                hidden = block_hidden(hidden)
        return backbone.final_norm(hidden)


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _foundation_state(path, run):
    state = load_state(path)
    if state.get("model_name", run.architecture) != run.architecture:
        raise ValueError(f"Expected the {run.architecture} Phase 1 foundation.")
    tokens = state.get("tokens_seen")
    expected = 700_000_000 if run.foundation.endswith("700M") else 100_000_000
    if tokens is not None and abs(tokens - expected) > 100_000:
        raise ValueError(f"Expected the {run.foundation} snapshot, found {tokens} tokens.")
    return state["model_state"]


def build_ner(run, pad_id, *, foundation=None, student_initialization=None,
              seed=None, device="cpu"):
    """Construct the exact BIO architecture and optionally load its training initialization."""
    student = run.method in {"student", "distilled"}
    if student:
        # The notebook constructed a full Phase 1 backbone before the smaller one.
        # Preserve that initialization order when recreating the seeded BIO head.
        source_model = build_model(run.architecture) if foundation else None
        backbone = EfficientGQATransformer(
            vocab_size=32000, max_seq_len=1280, n_layers=len(STUDENT_LAYERS)
        )
    else:
        backbone = build_model(run.architecture)
    if foundation:
        if student:
            # The archived initialization copied layers from Phase 1, before NER fine-tuning.
            source_model.load_state_dict(_foundation_state(foundation, run), strict=True)
            source = source_model.state_dict()
            initialized = backbone.state_dict()
            for name in initialized:
                source_name = name
                if name.startswith("blocks."):
                    _, index, suffix = name.split(".", 2)
                    source_name = f"blocks.{STUDENT_LAYERS[int(index)]}.{suffix}"
                initialized[name] = source[source_name]
            backbone.load_state_dict(initialized, strict=True)
        else:
            backbone.load_state_dict(_foundation_state(foundation, run), strict=True)

    if run.method == "lorma":
        # The original moved the frozen backbone before constructing adapters.
        backbone.to(device)
        if seed is not None:
            torch.manual_seed(seed + run.init_offset)
    model = LegalNER(backbone, run.architecture, run.method, pad_id).to(device)
    if run.method == "lora" and seed is not None:
        torch.manual_seed(seed + run.init_offset)
        model.classifier.reset_parameters()
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.reset_adapter()
    if student_initialization:
        if not student:
            raise ValueError("A student initialization applies only to seven-layer GQA.")
        state = load_state(student_initialization)
        _check_labels(state)
        if tuple(state["source_layers"]) != STUDENT_LAYERS:
            raise ValueError("Unexpected student source layers.")
        model.load_state_dict(state["model_state"], strict=True)
    return model


def _check_labels(state):
    if "labels" in state and tuple(state["labels"]) != BIO:
        raise ValueError("Checkpoint BIO label order differs from the Phase 2 schema.")


def restore_ner(model, state):
    """Read historical full-model, classifier-only, and adapter-only checkpoint schemas."""
    _check_labels(state)
    if model.method == "probe":
        model.classifier.load_state_dict(state["classifier_state"], strict=True)
    elif model.method in {"lora", "lorma"}:
        weights = state["trainable_state"]
        expected = {name for name, param in model.named_parameters() if param.requires_grad}
        if weights.keys() != expected:
            raise ValueError("Adapter checkpoint tensor names do not match trainable parameters.")
        model.load_state_dict(weights, strict=False)
    else:
        weights = state["model_state"] if "model_state" in state else state["model"]
        model.load_state_dict(weights, strict=True)


def ner_weights(model):
    if model.method == "probe":
        return {"classifier_state": model.classifier.state_dict()}
    if model.method in {"lora", "lorma"}:
        return {"trainable_state": {name: param.detach()
                                   for name, param in model.named_parameters()
                                   if param.requires_grad}}
    return {"model_state": model.state_dict()}
