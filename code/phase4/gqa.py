"""The original E01 tail-token Efficient GQA classifier."""

import torch
from torch import nn


class CJPEGQAClassifier(nn.Module):
    def __init__(self, backbone, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 2))
        nn.init.normal_(self.classifier[1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier[1].bias)

    def forward(self, input_ids, lengths):
        hidden, _ = self.backbone.hidden_states(input_ids)
        batch_size = input_ids.shape[0]
        row_indices = torch.arange(batch_size, device=input_ids.device)
        final_indices = lengths.long() - 1
        document_embeddings = hidden[row_indices, final_indices, :]
        logits = self.classifier(document_embeddings)
        return logits
