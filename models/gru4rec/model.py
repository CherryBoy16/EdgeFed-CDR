"""
GRU4Rec Model

This module implements the GRU4Rec architecture for
next-item recommendation.

Architecture:
Input Sequence
      ↓
Embedding Layer
      ↓
Packed GRU
      ↓
Dropout
      ↓
Fully Connected Layer
      ↓
Movie Scores (Logits)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from models.gru4rec.config import (
    EMBEDDING_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    PADDING_IDX,
)


class GRU4Rec(nn.Module):

    def __init__(self, num_items):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=PADDING_IDX
        )

        self.gru = nn.GRU(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.fc = nn.Linear(
            HIDDEN_DIM,
            num_items
        )

    def forward(self, sequences, lengths):

        embedded = self.embedding(sequences)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, hidden = self.gru(packed)

        hidden = hidden[-1]

        hidden = self.dropout(hidden)

        logits = self.fc(hidden)

        return logits
