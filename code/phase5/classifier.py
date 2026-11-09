"""Checkpoint-compatible EOS readout and document pooling."""

import torch
from torch import nn


class StatuteClassifier(nn.Module):
    def __init__(self, backbone, architecture, num_labels=100, dropout=.10):
        super().__init__()
        self.backbone = backbone
        self.architecture = architecture
        self.hidden_size = backbone.token_embedding.embedding_dim
        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.chunk_head = nn.Linear(self.hidden_size, num_labels)
        self.pool_beta = 2.0

    def window_repr(self, inputs, lengths):
        if self.architecture in ("prefix_lm", "encoder_decoder"):
            vectors = []
            for tokens, length in zip(inputs, lengths.tolist()):
                if length == 0:
                    vectors.append(self.chunk_head.weight.new_zeros(self.hidden_size))
                    continue
                unpadded = tokens[None, :length]
                if self.architecture == "encoder_decoder":
                    hidden = self.backbone.encode(unpadded)
                else:
                    hidden, _ = self.backbone.hidden_states(unpadded, prefix_length=length)
                vectors.append(hidden[0, -1])
            return torch.stack(vectors)
        hidden, _ = self.backbone.hidden_states(inputs)
        indices = torch.arange(len(inputs), device=inputs.device)
        return hidden[indices, lengths.clamp(min=1) - 1]

    def forward(self, inputs, lengths):
        batch, windows, width = inputs.shape
        vectors = self.window_repr(inputs.reshape(-1, width), lengths.flatten())
        vectors = vectors.reshape(batch, windows, -1)
        scores = self.chunk_head(self.dropout(self.norm(vectors))).float()
        active = lengths > 2
        scores = scores.masked_fill(~active[..., None], -1e5)
        count = active.sum(dim=1).clamp(min=1)
        return (torch.logsumexp(self.pool_beta * scores, dim=1)
                - torch.log(count[:, None])) / self.pool_beta
