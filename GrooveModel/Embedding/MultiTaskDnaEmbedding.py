from collections import OrderedDict
from dataclasses import dataclass, MISSING, field

import torch
from torch import nn

from GrooveModel.Embedding.Embedding import DnaEmbedding
from GrooveModel.Utils.SpecialTokens import SpecialTokens
from GrooveModel.Vocab import INSTRUMENT_VOCAB_SIZE, VELOCITY_VOCAB_SIZE, BEAT_UNIT_ABSOLUTE_VOCAB_SIZE, \
    BEAT_UNIT_RELATIVE_VOCAB_SIZE, OFFSET_VOCAB_SIZE, GRID_FACTOR_VOCAB_SIZE, BPM_VOCAB_SIZE, TIME_SIGNATURE_VOCAB_SIZE


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


class MultiTaskDNAEmbedding(DnaEmbedding):
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
            'grid_factor': nn.Embedding(GRID_FACTOR_VOCAB_SIZE, embedding_config.grid_factor.embedding_dim,
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
        feature_names = list(self.sub_embeddings.keys())
        embeddings = []

        for i, name in enumerate(feature_names):
            x = token[:, :, i]
            emb_layer = self.sub_embeddings[name]

            if x.min() < 0 or x.max() >= emb_layer.num_embeddings:
                raise ValueError(
                    f"Feature '{name}' has out-of-range indices: "
                    f"min={x.min().item()}, max={x.max().item()}, "
                    f"allowed=[0, {emb_layer.num_embeddings - 1}]"
                )

            embeddings.append(emb_layer(x))

        return self.normalize_embedding(torch.cat(embeddings, dim=-1))
