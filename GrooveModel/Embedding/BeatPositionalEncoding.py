import math

import torch
from torch import nn

from Configs import MAX_SEQUENCE_LENGTH


class BeatPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=MAX_SEQUENCE_LENGTH):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Precompute sinusoidal encoding table
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # shape: (max_len, d_model)

    def forward(self, x, beat_unit=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            beat_unit: Optional LongTensor of shape (batch_size, seq_len)
                       representing a position-like index for each token (e.g., beats)
        Returns:
            Tensor with beat-based positional encoding added
        """
        if beat_unit is None:
            seq_len = x.size(1)
            beat_unit = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)

        pos_emb = self.pe[beat_unit]  # shape: (batch_size, seq_len, d_model)
        return x + pos_emb
