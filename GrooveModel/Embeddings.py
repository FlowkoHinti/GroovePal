import math
import torch

from torch import nn
from omegaconf import DictConfig

class MultiDimDNAEmbedding(nn.Module):
    def __init__(self, config):
        super(MultiDimDNAEmbedding, self).__init__()

        self.embedding_dim = sum([
            config.embeddings.instruments.embedding_dim,
            config.embeddings.velocities.embedding_dim,
            config.embeddings.offsets.embedding_dim,
            config.embeddings.time_signature.embedding_dim,
            config.embeddings.grid_factor.embedding_dim,
            config.embeddings.bpm.embedding_dim
        ])

        self.instrument_embedding = nn.Embedding(
            config.embeddings.instruments.vocab_size,
            config.embeddings.instruments.embedding_dim
        )

        self.velocity_embedding = nn.Embedding(
            config.embeddings.velocities.vocab_size,
            config.embeddings.velocities.embedding_dim
        )

        self.offset_embedding = nn.Embedding(
            config.embeddings.offsets.vocab_size,
            config.embeddings.offsets.embedding_dim
        )

        self.time_signature_embedding = nn.Embedding(
            config.embeddings.time_signature.vocab_size,
            config.embeddings.time_signature.embedding_dim
        )

        self.grid_embedding = nn.Embedding(
            config.embeddings.grid_factor.vocab_size,
            config.embeddings.grid_factor.embedding_dim
        )

        self.bpm_embedding = nn.Embedding(
            config.embeddings.bpm.vocab_size,
            config.embeddings.bpm.embedding_dim
        )

    def forward(self, token: torch.Tensor):
        """
        token: Tensor of shape (batch_size, seq_len, 9)
        Each token has 9 fields, ordered as:
        [instrument, velocity, beat_unit, beat_unit_offset, grid_factor, bpm, time_signature, number_of_bars, ticks_p_qn]
        """

        instrument_token_embedding = self.instrument_embedding(token[:, :, 0])
        velocity_token_embedding = self.velocity_embedding(token[:, :, 1])
        offset_token_embedding = self.offset_embedding(token[:, :, 3])
        grid_token_embedding = self.grid_embedding(token[:, :, 4])
        bpm_token_embedding = self.bpm_embedding(token[:, :, 5])
        time_signature_token_embedding = self.time_signature_embedding(token[:, :, 6])

        # Concatenate embeddings across the last dimension
        token_embedding = torch.cat([
            instrument_token_embedding,
            velocity_token_embedding,
            offset_token_embedding,
            time_signature_token_embedding,
            grid_token_embedding,
            bpm_token_embedding
        ], dim=-1)

        return token_embedding



class BeatPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.d_model = d_model

        # Precompute sinusoidal encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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