import math
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
from omegaconf import MISSING
from torch import nn

from Configs import MAX_SEQUENCE_LENGTH
from GrooveModel.Utils.SpecialTokens import SpecialTokens
from GrooveModel.Vocab import INSTRUMENT_VOCAB_SIZE, VELOCITY_VOCAB_SIZE, OFFSET_VOCAB_SIZE, TIME_SIGNATURE_VOCAB_SIZE, \
    GRID_FACTOR_VOCAB_SIZE, BPM_VOCAB_SIZE, BEAT_UNIT_ABSOLUTE_VOCAB_SIZE, BEAT_UNIT_RELATIVE_VOCAB_SIZE


@dataclass
class SubEmbeddingConfig:
    embedding_dim: int = MISSING


@dataclass
class BeatUnitsConfig:
    embedding_dim: int = MISSING
    absolute_beat_units: bool = MISSING


@dataclass
class MultiTaskDNAEmbeddingConfig:
    instruments: SubEmbeddingConfig = field(default_factory=SubEmbeddingConfig)
    velocities: SubEmbeddingConfig = field(default_factory=SubEmbeddingConfig)
    offsets: SubEmbeddingConfig = field(default_factory=SubEmbeddingConfig)
    time_signature: SubEmbeddingConfig = field(default_factory=SubEmbeddingConfig)
    grid_factor: SubEmbeddingConfig = field(default_factory=SubEmbeddingConfig)
    bpm: SubEmbeddingConfig = field(default_factory=SubEmbeddingConfig)
    beat_units: BeatUnitsConfig = field(default_factory=BeatUnitsConfig)
    normalize_embeddings: bool = False


class MultiTaskDNAEmbedding(nn.Module):
    def __init__(self, embedding_config: MultiTaskDNAEmbeddingConfig):
        super(MultiTaskDNAEmbedding, self).__init__()

        self.normalize_embedding_flag = embedding_config.normalize_embeddings
        self.absolute_beat_units = embedding_config.beat_units.absolute_beat_units

        self.sub_embeddings = nn.ModuleDict(OrderedDict({
            'instrument': nn.Embedding(INSTRUMENT_VOCAB_SIZE, embedding_config.instruments.embedding_dim,
                                       padding_idx=SpecialTokens.PAD),
            'velocity': nn.Embedding(VELOCITY_VOCAB_SIZE, embedding_config.velocities.embedding_dim,
                                     padding_idx=SpecialTokens.PAD),
            'beat_unit': nn.Embedding(
                BEAT_UNIT_ABSOLUTE_VOCAB_SIZE if self.absolute_beat_units else BEAT_UNIT_RELATIVE_VOCAB_SIZE,
                embedding_config.beat_units.embedding_dim,
                padding_idx=SpecialTokens.PAD
            ),
            'offset': nn.Embedding(OFFSET_VOCAB_SIZE, embedding_config.offsets.embedding_dim,
                                   padding_idx=SpecialTokens.PAD),
            'grid': nn.Embedding(GRID_FACTOR_VOCAB_SIZE, embedding_config.grid_factor.embedding_dim,
                                 padding_idx=SpecialTokens.PAD),
            'bpm': nn.Embedding(BPM_VOCAB_SIZE, embedding_config.bpm.embedding_dim, padding_idx=SpecialTokens.PAD),
            'time_signature': nn.Embedding(TIME_SIGNATURE_VOCAB_SIZE, embedding_config.time_signature.embedding_dim,
                                           padding_idx=SpecialTokens.PAD)
        }))

        # Store embedding dimension as integer attribute
        self._embedding_dim = sum(e.embedding_dim for e in [
            embedding_config.instruments,
            embedding_config.velocities,
            embedding_config.beat_units,
            embedding_config.offsets,
            embedding_config.time_signature,
            embedding_config.grid_factor,
            embedding_config.bpm
        ])

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.normalize_embedding_flag:
            return nn.functional.layer_norm(embedding, embedding.shape[-1:])
        return embedding

    def forward(self, token: torch.Tensor):
        # Assume fixed order of features in token shape: (batch, seq, feature_index)
        feature_names = list(self.sub_embeddings.keys())

        embeddings = [
            self.sub_embeddings[name](token[:, :, i])
            for i, name in enumerate(feature_names)
        ]

        token_embedding = torch.cat(embeddings, dim=-1)
        return self.normalize_embedding(token_embedding)


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
