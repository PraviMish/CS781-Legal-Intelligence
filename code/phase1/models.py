"""Phase-1 legal language model architectures.

Architectural definitions are derived from the final executed Phase-1 notebook.
Parameter/module names are deliberately kept unchanged for checkpoint compatibility.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

# Original legal BPE special-token order: <pad>=0, <bos>=1, <eos>=2, <unk>=3.
BOS_ID = 1

class SelfAttention(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
    ):

        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(
            d_model,
            3 * d_model,
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model,
        )


    def forward(
        self,
        x,
        causal=False,
        attention_mask=None,
        past_kv=None,
        use_cache=False,
    ):

        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)


        def reshape_heads(tensor):

            return (
                tensor.view(
                    B,
                    T,
                    self.n_heads,
                    self.head_dim,
                )
                .transpose(1, 2)
            )


        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)


        past_length = 0

        if past_kv is not None:

            past_k, past_v = past_kv

            past_length = past_k.size(2)

            k = torch.cat(
                [past_k, k],
                dim=2,
            )

            v = torch.cat(
                [past_v, v],
                dim=2,
            )


        new_cache = (k, v) if use_cache else None


        # During autoregressive cached decoding T=1.
        # The new token may attend to every cached previous token.
        if past_length > 0 and T == 1:

            is_causal = False
            attention_mask = None

        else:

            is_causal = (
                causal
                and attention_mask is None
            )


        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=is_causal,
            dropout_p=0.0,
        )


        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        y = self.out_proj(y)

        return y, new_cache

class StandardMLP(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
        dropout=0.1,
    ):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )


    def forward(self, x):
        return self.net(x)


class StandardTransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout=0.1,
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attention = SelfAttention(
            d_model,
            n_heads,
        )

        self.mlp = StandardMLP(
            d_model,
            d_ff,
            dropout,
        )

        self.dropout = nn.Dropout(dropout)


    def forward(
        self,
        x,
        causal,
        attention_mask=None,
        past_kv=None,
        use_cache=False,
    ):

        attn_output, new_cache = self.attention(
            self.norm1(x),
            causal=causal,
            attention_mask=attention_mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )

        x = x + self.dropout(attn_output)

        x = x + self.mlp(
            self.norm2(x)
        )

        return x, new_cache

class DecoderOnlyTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_seq_len=1024,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        dropout=0.1,
    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                StandardTransformerBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        # Weight tying.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)


    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )


    def hidden_states(
        self,
        input_ids,
        past_cache=None,
        use_cache=False,
    ):

        B, T = input_ids.shape

        past_length = 0

        if (
            past_cache is not None
            and past_cache[0] is not None
        ):

            past_length = (
                past_cache[0][0].size(2)
            )

        if past_length + T > self.max_seq_len:

            raise ValueError(
                "Sequence exceeds model context length."
            )


        positions = torch.arange(
            past_length,
            past_length + T,
            device=input_ids.device,
        )

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)[None, :, :]
        )

        x = self.dropout(x)

        new_cache = []

        for i, block in enumerate(self.blocks):

            layer_cache = (
                None
                if past_cache is None
                else past_cache[i]
            )

            x, cache = block(
                x,
                causal=True,
                past_kv=layer_cache,
                use_cache=use_cache,
            )

            if use_cache:
                new_cache.append(cache)

        x = self.final_norm(x)

        return x, new_cache if use_cache else None


    def forward(
        self,
        input_ids,
        past_cache=None,
        use_cache=False,
        last_only=False,
    ):

        hidden, new_cache = self.hidden_states(
            input_ids,
            past_cache=past_cache,
            use_cache=use_cache,
        )

        if last_only:
            hidden = hidden[:, -1:, :]

        logits = self.lm_head(hidden)

        return logits, new_cache


    def loss_on_block(self, input_ids):

        logits, _ = self(input_ids)

        prediction_logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]

        return F.cross_entropy(
            prediction_logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
        )

class PrefixLMTransformer(DecoderOnlyTransformer):

    def build_prefix_mask(
        self,
        sequence_length,
        prefix_length,
        device,
    ):
        """
        Prefix tokens attend bidirectionally within the prefix.

        Continuation tokens attend to:
          - every prefix token
          - previous continuation tokens
          - themselves
        """

        positions = torch.arange(
            sequence_length,
            device=device,
        )

        query = positions[:, None]
        key = positions[None, :]

        prefix_queries = (
            query < prefix_length
        )

        prefix_keys = (
            key < prefix_length
        )

        prefix_to_prefix = (
            prefix_queries
            & prefix_keys
        )

        continuation_queries = (
            query >= prefix_length
        )

        causal_continuation = (
            continuation_queries
            & (key <= query)
        )

        allowed = (
            prefix_to_prefix
            | causal_continuation
        )

        return allowed


    def hidden_states(
        self,
        input_ids,
        prefix_length=None,
        past_cache=None,
        use_cache=False,
    ):

        B, T = input_ids.shape

        past_length = 0

        if (
            past_cache is not None
            and past_cache[0] is not None
        ):

            past_length = (
                past_cache[0][0].size(2)
            )


        if past_length + T > self.max_seq_len:

            raise ValueError(
                "Sequence exceeds Prefix-LM context length."
            )


        positions = torch.arange(
            past_length,
            past_length + T,
            device=input_ids.device,
        )


        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)[None, :, :]
        )

        x = self.dropout(x)


        # Training case:
        # prefix exists and continuation already exists in same sequence.
        attention_mask = None
        causal = False

        if (
            past_length == 0
            and prefix_length is not None
            and prefix_length < T
        ):

            attention_mask = self.build_prefix_mask(
                T,
                prefix_length,
                input_ids.device,
            )


        # Initial generation pass:
        # the complete prompt is the prefix and can be bidirectional.
        # Therefore causal=False and attention_mask=None.


        new_cache = []

        for i, block in enumerate(self.blocks):

            layer_cache = (
                None
                if past_cache is None
                else past_cache[i]
            )

            x, cache = block(
                x,
                causal=causal,
                attention_mask=attention_mask,
                past_kv=layer_cache,
                use_cache=use_cache,
            )

            if use_cache:
                new_cache.append(cache)


        x = self.final_norm(x)

        return x, new_cache if use_cache else None


    def forward(
        self,
        input_ids,
        prefix_length=None,
        past_cache=None,
        use_cache=False,
        last_only=False,
    ):

        hidden, new_cache = self.hidden_states(
            input_ids,
            prefix_length=prefix_length,
            past_cache=past_cache,
            use_cache=use_cache,
        )

        if last_only:
            hidden = hidden[:, -1:, :]

        logits = self.lm_head(hidden)

        return logits, new_cache


    def loss_on_block(self, input_ids):

        T = input_ids.size(1)

        prefix_length = int(0.75 * T)

        hidden, _ = self.hidden_states(
            input_ids,
            prefix_length=prefix_length,
        )

        # Position prefix_length - 1 predicts the first continuation token.
        prediction_hidden = hidden[
            :,
            prefix_length - 1 : -1,
            :
        ]

        labels = input_ids[
            :,
            prefix_length:
        ]

        logits = self.lm_head(
            prediction_hidden
        )

        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
        )

class CrossAttention(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
    ):

        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(
            d_model,
            d_model,
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model,
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model,
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model,
        )


    def forward(
        self,
        x,
        memory,
        cached_memory_kv=None,
        use_cache=False,
    ):

        B, T, C = x.shape

        q = self.q_proj(x)

        q = (
            q.view(
                B,
                T,
                self.n_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )


        if cached_memory_kv is None:

            S = memory.size(1)

            k = self.k_proj(memory)
            v = self.v_proj(memory)

            k = (
                k.view(
                    B,
                    S,
                    self.n_heads,
                    self.head_dim,
                )
                .transpose(1, 2)
            )

            v = (
                v.view(
                    B,
                    S,
                    self.n_heads,
                    self.head_dim,
                )
                .transpose(1, 2)
            )

        else:

            k, v = cached_memory_kv


        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            dropout_p=0.0,
        )


        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        y = self.out_proj(y)

        new_memory_cache = (
            (k, v)
            if use_cache
            else None
        )

        return y, new_memory_cache

class EncoderBlock(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout,
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attention = SelfAttention(
            d_model,
            n_heads,
        )

        self.mlp = StandardMLP(
            d_model,
            d_ff,
            dropout,
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        y, _ = self.attention(
            self.norm1(x),
            causal=False,
        )

        x = x + self.dropout(y)

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


class DecoderBlock(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout,
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attention = SelfAttention(
            d_model,
            n_heads,
        )

        self.cross_attention = CrossAttention(
            d_model,
            n_heads,
        )

        self.mlp = StandardMLP(
            d_model,
            d_ff,
            dropout,
        )

        self.dropout = nn.Dropout(dropout)


    def forward(
        self,
        x,
        memory,
        self_cache=None,
        memory_cache=None,
        use_cache=False,
    ):

        y, new_self_cache = self.self_attention(
            self.norm1(x),
            causal=True,
            past_kv=self_cache,
            use_cache=use_cache,
        )

        x = x + self.dropout(y)


        y, new_memory_cache = self.cross_attention(
            self.norm2(x),
            memory,
            cached_memory_kv=memory_cache,
            use_cache=use_cache,
        )

        x = x + self.dropout(y)

        x = x + self.mlp(
            self.norm3(x)
        )


        return (
            x,
            new_self_cache,
            new_memory_cache,
        )


class EncoderDecoderTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        encoder_max_len=1024,
        decoder_max_len=384,
        d_model=640,
        n_encoder_layers=7,
        n_decoder_layers=7,
        n_heads=10,
        d_ff=2560,
        dropout=0.1,
    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.encoder_position = nn.Embedding(
            encoder_max_len,
            d_model,
        )

        self.decoder_position = nn.Embedding(
            decoder_max_len,
            d_model,
        )


        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                )
                for _ in range(n_encoder_layers)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                )
                for _ in range(n_decoder_layers)
            ]
        )


        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)


    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )


    def encode(self, source_ids):

        B, S = source_ids.shape

        if S > self.encoder_max_len:
            raise ValueError(
                "Encoder sequence too long."
            )

        positions = torch.arange(
            S,
            device=source_ids.device,
        )

        x = (
            self.token_embedding(source_ids)
            + self.encoder_position(positions)[None, :, :]
        )

        x = self.dropout(x)

        for block in self.encoder_blocks:
            x = block(x)

        return self.encoder_norm(x)


    def decode(
        self,
        decoder_ids,
        memory,
        self_caches=None,
        memory_caches=None,
        use_cache=False,
        last_only=False,
    ):

        B, T = decoder_ids.shape

        past_length = 0

        if (
            self_caches is not None
            and self_caches[0] is not None
        ):

            past_length = (
                self_caches[0][0].size(2)
            )


        if past_length + T > self.decoder_max_len:
            raise ValueError(
                "Decoder sequence too long."
            )


        positions = torch.arange(
            past_length,
            past_length + T,
            device=decoder_ids.device,
        )

        x = (
            self.token_embedding(decoder_ids)
            + self.decoder_position(positions)[None, :, :]
        )

        x = self.dropout(x)


        new_self_caches = []
        new_memory_caches = []


        for i, block in enumerate(self.decoder_blocks):

            self_cache = (
                None
                if self_caches is None
                else self_caches[i]
            )

            memory_cache = (
                None
                if memory_caches is None
                else memory_caches[i]
            )

            (
                x,
                new_self_cache,
                new_memory_cache,
            ) = block(
                x,
                memory,
                self_cache=self_cache,
                memory_cache=memory_cache,
                use_cache=use_cache,
            )

            if use_cache:

                new_self_caches.append(
                    new_self_cache
                )

                new_memory_caches.append(
                    new_memory_cache
                )


        x = self.decoder_norm(x)

        if last_only:
            x = x[:, -1:, :]

        logits = self.lm_head(x)


        return (
            logits,
            new_self_caches if use_cache else None,
            new_memory_caches if use_cache else None,
        )


    def loss_on_block(self, input_ids):

        T = input_ids.size(1)

        source_length = int(
            0.75 * T
        )

        source_ids = input_ids[
            :,
            :source_length
        ]

        target_ids = input_ids[
            :,
            source_length:
        ]


        memory = self.encode(
            source_ids
        )


        bos_column = torch.full(
            (
                input_ids.size(0),
                1,
            ),
            BOS_ID,
            dtype=torch.long,
            device=input_ids.device,
        )


        decoder_input = torch.cat(
            [
                bos_column,
                target_ids[:, :-1],
            ],
            dim=1,
        )


        logits, _, _ = self.decode(
            decoder_input,
            memory,
        )


        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_ids.reshape(-1),
        )

class RMSNorm(nn.Module):

    def __init__(
        self,
        dim,
        eps=1e-6,
    ):

        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

        self.eps = eps


    def forward(self, x):

        rms = x.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        x = x * torch.rsqrt(
            rms + self.eps
        )

        return self.weight * x


def rotate_half(x):

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    return torch.stack(
        (-x2, x1),
        dim=-1,
    ).flatten(-2)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim,
        max_seq_len=1024,
        base=10_000,
    ):

        super().__init__()

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                ).float()
                / head_dim
            )
        )

        positions = torch.arange(
            max_seq_len
        ).float()

        frequencies = torch.outer(
            positions,
            inv_freq,
        )

        emb = torch.repeat_interleave(
            frequencies,
            2,
            dim=-1,
        )

        self.register_buffer(
            "cos",
            emb.cos(),
            persistent=False,
        )

        self.register_buffer(
            "sin",
            emb.sin(),
            persistent=False,
        )


    def apply_rotary(
        self,
        q,
        k,
        start_position,
    ):

        T = q.size(2)

        cos = self.cos[
            start_position:
            start_position + T
        ][None, None, :, :]

        sin = self.sin[
            start_position:
            start_position + T
        ][None, None, :, :]

        q = (
            q * cos
            + rotate_half(q) * sin
        )

        k = (
            k * cos
            + rotate_half(k) * sin
        )

        return q, k

class GroupedQueryAttention(nn.Module):

    def __init__(
        self,
        d_model,
        n_query_heads,
        n_kv_heads,
        max_seq_len,
    ):

        super().__init__()

        assert d_model % n_query_heads == 0
        assert n_query_heads % n_kv_heads == 0

        self.d_model = d_model
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads

        self.head_dim = (
            d_model // n_query_heads
        )

        self.kv_dim = (
            n_kv_heads * self.head_dim
        )


        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.k_proj = nn.Linear(
            d_model,
            self.kv_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            d_model,
            self.kv_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )


        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len=max_seq_len,
        )


    def forward(
        self,
        x,
        past_kv=None,
        use_cache=False,
    ):

        B, T, C = x.shape

        past_length = 0

        if past_kv is not None:
            past_length = past_kv[0].size(2)


        q = self.q_proj(x).view(
            B,
            T,
            self.n_query_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = self.k_proj(x).view(
            B,
            T,
            self.n_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = self.v_proj(x).view(
            B,
            T,
            self.n_kv_heads,
            self.head_dim,
        ).transpose(1, 2)


        q, k = self.rope.apply_rotary(
                q,
                k,
                past_length,
        )


        if past_kv is not None:

            old_k, old_v = past_kv

            k = torch.cat(
                [old_k, k],
                dim=2,
            )

            v = torch.cat(
                [old_v, v],
                dim=2,
            )


        new_cache = (
            (k, v)
            if use_cache
            else None
        )


        # PyTorch supports native GQA on recent CUDA builds.
        # If unavailable, repeat K/V heads explicitly.
        try:

            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=(
                    past_length == 0
                    and T > 1
                ),
                dropout_p=0.0,
                enable_gqa=True,
            )

        except TypeError:

            repetitions = (
                self.n_query_heads
                // self.n_kv_heads
            )

            expanded_k = k.repeat_interleave(
                repetitions,
                dim=1,
            )

            expanded_v = v.repeat_interleave(
                repetitions,
                dim=1,
            )

            y = F.scaled_dot_product_attention(
                q,
                expanded_k,
                expanded_v,
                is_causal=(
                    past_length == 0
                    and T > 1
                ),
                dropout_p=0.0,
            )


        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        return self.out_proj(y), new_cache

class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
    ):

        super().__init__()

        self.gate = nn.Linear(
            d_model,
            d_ff,
            bias=False,
        )

        self.up = nn.Linear(
            d_model,
            d_ff,
            bias=False,
        )

        self.down = nn.Linear(
            d_ff,
            d_model,
            bias=False,
        )


    def forward(self, x):

        return self.down(
            F.silu(self.gate(x))
            * self.up(x)
        )


class EfficientBlock(nn.Module):

    def __init__(
        self,
        d_model,
        n_query_heads,
        n_kv_heads,
        d_ff,
        max_seq_len,
    ):

        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.attention = GroupedQueryAttention(
            d_model,
            n_query_heads,
            n_kv_heads,
            max_seq_len,
        )

        self.mlp = SwiGLU(
            d_model,
            d_ff,
        )


    def forward(
        self,
        x,
        past_kv=None,
        use_cache=False,
    ):

        y, new_cache = self.attention(
            self.norm1(x),
            past_kv=past_kv,
            use_cache=use_cache,
        )

        x = x + y

        x = x + self.mlp(
            self.norm2(x)
        )

        return x, new_cache


class EfficientGQATransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_seq_len=1024,
        d_model=768,
        n_layers=14,
        n_query_heads=12,
        n_kv_heads=4,
        d_ff=2048,
    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.blocks = nn.ModuleList(
            [
                EfficientBlock(
                    d_model,
                    n_query_heads,
                    n_kv_heads,
                    d_ff,
                    max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = RMSNorm(
            d_model
        )

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(self._init_weights)


    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )


    def hidden_states(
        self,
        input_ids,
        past_cache=None,
        use_cache=False,
    ):

        B, T = input_ids.shape

        past_length = 0

        if (
            past_cache is not None
            and past_cache[0] is not None
        ):
            past_length = past_cache[0][0].size(2)


        if past_length + T > self.max_seq_len:

            raise ValueError(
                "Sequence exceeds model context."
            )


        x = self.token_embedding(
            input_ids
        )

        new_cache = []

        for i, block in enumerate(self.blocks):

            layer_cache = (
                None
                if past_cache is None
                else past_cache[i]
            )

            x, cache = block(
                x,
                past_kv=layer_cache,
                use_cache=use_cache,
            )

            if use_cache:
                new_cache.append(cache)


        x = self.final_norm(x)

        return x, new_cache if use_cache else None


    def forward(
        self,
        input_ids,
        past_cache=None,
        use_cache=False,
        last_only=False,
    ):

        hidden, new_cache = self.hidden_states(
            input_ids,
            past_cache=past_cache,
            use_cache=use_cache,
        )

        if last_only:
            hidden = hidden[:, -1:, :]

        logits = self.lm_head(hidden)

        return logits, new_cache


    def loss_on_block(self, input_ids):

        hidden, _ = self.hidden_states(
            input_ids
        )

        logits = self.lm_head(
            hidden[:, :-1, :]
        )

        labels = input_ids[:, 1:]

        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
        )



MODEL_NAMES = ("decoder_only", "encoder_decoder", "prefix_lm", "efficient_gqa")

def build_model(name: str, vocab_size: int = 32000, context_length: int = 1280) -> nn.Module:
    """Construct one of the four fixed, historically submitted architectures.

    Defaults correspond to the original competition runs, not a new sweep.
    """
    if name == "decoder_only":
        return DecoderOnlyTransformer(vocab_size=vocab_size, max_seq_len=context_length,
                                      d_model=768, n_layers=12, n_heads=12, d_ff=3072, dropout=0.1)
    if name == "encoder_decoder":
        return EncoderDecoderTransformer(vocab_size=vocab_size, encoder_max_len=1024,
                                         decoder_max_len=640, d_model=640,
                                         n_encoder_layers=7, n_decoder_layers=7,
                                         n_heads=10, d_ff=2560, dropout=0.1)
    if name == "prefix_lm":
        return PrefixLMTransformer(vocab_size=vocab_size, max_seq_len=context_length,
                                   d_model=768, n_layers=12, n_heads=12, d_ff=3072, dropout=0.1)
    if name == "efficient_gqa":
        return EfficientGQATransformer(vocab_size=vocab_size, max_seq_len=context_length,
                                       d_model=768, n_layers=14, n_query_heads=12,
                                       n_kv_heads=4, d_ff=2048)
    raise ValueError(f"Unknown architecture: {name}")

def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
