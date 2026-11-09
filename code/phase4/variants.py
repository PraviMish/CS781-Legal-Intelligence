"""Phase 4 classifiers and adaptation objectives from the research notebook."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class E02HierarchicalHead(nn.Module):
    """Predict an attention-pooled correction to the frozen tail logits."""

    def __init__(self, hidden_dim=768, num_chunks=5, dropout=0.15):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(num_chunks, hidden_dim))
        self.attention = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.correction = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 2)
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, features, chunk_logits, mask, return_attention=False):
        features = features.float()
        chunk_logits = chunk_logits.float()
        mask = mask.bool()
        batch_size = features.shape[0]
        assert mask.any(dim=1).all()
        normalized = self.norm(features)
        positioned = normalized + self.position_embedding.unsqueeze(0)
        scores = self.attention(positioned).squeeze(-1)
        scores = scores.masked_fill(~mask, -10000.0)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * normalized, dim=1)
        tail_idx = mask.long().sum(dim=1) - 1
        row_idx = torch.arange(batch_size, device=features.device)
        tail_embedding = normalized[row_idx, tail_idx]
        tail_logits = chunk_logits[row_idx, tail_idx]
        combined = torch.cat([pooled, tail_embedding, pooled - tail_embedding], dim=-1)
        correction = self.correction(combined)
        final_logits = tail_logits + correction
        if return_attention:
            return (final_logits, weights)
        return final_logits


class E03LoRMAPlusLinear(nn.Module):
    """Multiply the frozen projection output by a low-rank correction."""

    def __init__(self, base, rank=16, alpha=32):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        d_out = base.out_features
        self.lorma_A = nn.Parameter(torch.empty(rank, d_out))
        self.lorma_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lorma_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        z = F.linear(x, self.base.weight, bias=None)
        update = F.linear(F.linear(z, self.lorma_A), self.lorma_B)
        y = z + self.scale * update
        if self.base.bias is not None:
            y = y + self.base.bias
        return y


class E03GQALoRMAClassifier(nn.Module):
    def __init__(self, backbone, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.adapter_count = 0
        for block in self.backbone.blocks:
            for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                original = getattr(block.attention, name)
                adapted = E03LoRMAPlusLinear(original, rank=rank, alpha=alpha)
                setattr(block.attention, name, adapted)
                self.adapter_count += 1
            for name in ["gate", "up", "down"]:
                original = getattr(block.mlp, name)
                adapted = E03LoRMAPlusLinear(original, rank=rank, alpha=alpha)
                setattr(block.mlp, name, adapted)
                self.adapter_count += 1
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, 2))
        nn.init.normal_(self.classifier[1].weight, std=0.02)
        nn.init.zeros_(self.classifier[1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, input_ids, lengths):
        hidden, _ = self.backbone.hidden_states(input_ids)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        last_indices = lengths.long() - 1
        embeddings = hidden[batch_indices, last_indices]
        return self.classifier(embeddings)


class E04DecoderClassifier(nn.Module):
    def __init__(self, backbone, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        hidden_dim = backbone.d_model
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))
        nn.init.normal_(self.classifier[1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier[1].bias)

    def forward(self, input_ids, lengths):
        hidden, _ = self.backbone.hidden_states(input_ids)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        positions = lengths.long() - 1
        document_states = hidden[rows, positions]
        logits = self.classifier(document_states)
        return logits


class E05PrefixClassifier(nn.Module):
    """Classify the end of a bidirectional prefix with padding masked."""

    def __init__(self, backbone, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.d_model, 2))
        nn.init.normal_(self.classifier[1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier[1].bias)

    def prefix_hidden_states(self, input_ids, lengths):
        T = input_ids.shape[1]
        assert T <= self.backbone.max_seq_len
        positions = torch.arange(T, device=input_ids.device)
        x = (
            self.backbone.token_embedding(input_ids)
            + self.backbone.position_embedding(positions)[None, :, :]
        )
        x = self.backbone.dropout(x)
        valid_keys = torch.arange(T, device=input_ids.device)[None, :] < lengths[:, None]
        attention_mask = valid_keys[:, None, None, :]
        for block in self.backbone.blocks:
            x, _ = block(x, causal=False, attention_mask=attention_mask)
        x = self.backbone.final_norm(x)
        return x

    def forward(self, input_ids, lengths):
        hidden = self.prefix_hidden_states(input_ids, lengths)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        last_positions = lengths.long() - 1
        document_states = hidden[rows, last_positions]
        return self.classifier(document_states)


class E06EncoderDecoderClassifier(nn.Module):
    """Classify a single BOS query over padding-masked encoder memory."""

    def __init__(self, backbone, bos_id, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.bos_id = int(bos_id)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.d_model, 2))
        nn.init.normal_(self.classifier[1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier[1].bias)

    def encode_masked(self, input_ids, lengths):
        S = input_ids.shape[1]
        assert S <= self.backbone.encoder_max_len
        positions = torch.arange(S, device=input_ids.device)
        x = (
            self.backbone.token_embedding(input_ids)
            + self.backbone.encoder_position(positions)[None, :, :]
        )
        x = self.backbone.dropout(x)
        valid_keys = positions[None, :] < lengths[:, None]
        attention_mask = valid_keys[:, None, None, :]
        for block in self.backbone.encoder_blocks:
            y, _ = block.attention(block.norm1(x), causal=False, attention_mask=attention_mask)
            x = x + block.dropout(y)
            x = x + block.mlp(block.norm2(x))
        return self.backbone.encoder_norm(x)

    def masked_cross_attention(self, layer, query, memory, lengths):
        B, T, C = query.shape
        S = memory.shape[1]
        H = layer.n_heads
        D = layer.head_dim
        q = layer.q_proj(query)
        k = layer.k_proj(memory)
        v = layer.v_proj(memory)
        q = q.reshape(B, T, H, D).transpose(1, 2)
        k = k.reshape(B, S, H, D).transpose(1, 2)
        v = v.reshape(B, S, H, D).transpose(1, 2)
        valid_keys = torch.arange(S, device=query.device)[None, :] < lengths[:, None]
        attention_mask = valid_keys[:, None, None, :]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return layer.out_proj(y)

    def decoder_query(self, memory, lengths):
        B = memory.shape[0]
        decoder_ids = torch.full((B, 1), self.bos_id, dtype=torch.long, device=memory.device)
        positions = torch.zeros(1, dtype=torch.long, device=memory.device)
        x = (
            self.backbone.token_embedding(decoder_ids)
            + self.backbone.decoder_position(positions)[None, :, :]
        )
        x = self.backbone.dropout(x)
        for block in self.backbone.decoder_blocks:
            y, _ = block.self_attention(block.norm1(x), causal=True)
            x = x + block.dropout(y)
            y = self.masked_cross_attention(block.cross_attention, block.norm2(x), memory, lengths)
            x = x + block.dropout(y)
            x = x + block.mlp(block.norm3(x))
        return self.backbone.decoder_norm(x)[:, 0, :]

    def forward(self, input_ids, lengths):
        memory = self.encode_masked(input_ids, lengths)
        document_representation = self.decoder_query(memory, lengths)
        logits = self.classifier(document_representation)
        return logits


class E07AdaptiveLinear(nn.Module):
    """Apply a frozen projection plus masked, trainable rank components."""

    def __init__(self, base, rank=16, alpha=16):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        d_in = base.in_features
        d_out = base.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.empty(d_out, rank))
        self.lora_E = nn.Parameter(torch.zeros(rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
        self.register_buffer("rank_mask", torch.ones(rank))
        self.register_buffer("score_ema", torch.zeros(rank))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        original = self.base(x)
        low_rank = F.linear(x, self.lora_A)
        scaled = low_rank * (self.lora_E * self.rank_mask)
        update = F.linear(scaled, self.lora_B)
        return original + self.scale * update


class E07GQAClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.adapters = []
        for block in self.backbone.blocks:
            for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                original = getattr(block.attention, name)
                adapted = E07AdaptiveLinear(original, rank=16, alpha=16)
                setattr(block.attention, name, adapted)
                self.adapters.append(adapted)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 2))
        nn.init.normal_(self.classifier[1].weight, std=0.02)
        nn.init.zeros_(self.classifier[1].bias)

    def forward(self, input_ids, lengths):
        hidden, _ = self.backbone.hidden_states(input_ids)
        rows = torch.arange(len(input_ids), device=input_ids.device)
        pooled = hidden[rows, lengths.long() - 1]
        return self.classifier(pooled)


class E08StudentBlock(nn.Module):
    def __init__(self, dim=384, heads=6, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout_p = dropout
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = (
            self.qkv(self.norm1(x))
            .reshape(B, T, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().reshape(B, T, D)
        x = x + self.dropout(self.proj(y))
        x = x + self.mlp(self.norm2(x))
        return x


class E08SmallTransformer(nn.Module):
    def __init__(self, vocab_size=32000, dim=384, layers=6, heads=6, max_length=1024, dropout=0.1):
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_length, dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [E08StudentBlock(dim=dim, heads=heads, dropout=dropout) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 2)

    def forward(self, input_ids, lengths):
        B, T = input_ids.shape
        assert T <= self.max_length
        positions = torch.arange(T, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        rows = torch.arange(B, device=input_ids.device)
        pooled = x[rows, lengths.long() - 1]
        return self.classifier(pooled)


@torch.no_grad()
def update_importance(model):
    for adapter in model.adapters:
        if adapter.lora_E.grad is None:
            continue
        sensitivity = adapter.lora_E.grad.detach().float().abs() * (
            adapter.lora_E.detach().float().abs() + 0.001
        )
        adapter.score_ema.mul_(0.95).add_(sensitivity, alpha=0.05)


@torch.no_grad()
def allocate_ranks(model, budget, min_rank=2):
    """Retain a minimum per projection and allocate the remaining budget globally."""
    adapters = model.adapters
    n = len(adapters)
    active_now = sum(int(a.rank_mask.sum().item()) for a in adapters)
    budget = max(n * min_rank, min(budget, active_now))
    mandatory = []
    candidates = []
    for module_idx, adapter in enumerate(adapters):
        active = torch.nonzero(adapter.rank_mask > 0, as_tuple=False).flatten().tolist()
        ranked = sorted(active, key=lambda j: float(adapter.score_ema[j].item()), reverse=True)
        assert len(ranked) >= min_rank
        mandatory.extend((module_idx, j) for j in ranked[:min_rank])
        candidates.extend(
            (float(adapter.score_ema[j].item()), module_idx, j) for j in ranked[min_rank:]
        )
    candidates.sort(reverse=True)
    selected = mandatory + [(m, j) for _, m, j in candidates[: budget - len(mandatory)]]
    for adapter in adapters:
        adapter.rank_mask.zero_()
    for module_idx, rank_idx in selected:
        adapters[module_idx].rank_mask[rank_idx] = 1.0
    return sum(int(a.rank_mask.sum().item()) for a in adapters)


def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, kd_weight=0.65):
    T = temperature
    hard_loss = F.cross_entropy(student_logits.float(), labels)
    soft_loss = (
        F.kl_div(
            F.log_softmax(student_logits.float() / T, dim=-1),
            F.softmax(teacher_logits.float() / T, dim=-1),
            reduction="batchmean",
        )
        * T**2
    )
    total = (1.0 - kd_weight) * hard_loss + kd_weight * soft_loss
    return (total, hard_loss.detach(), soft_loss.detach())
