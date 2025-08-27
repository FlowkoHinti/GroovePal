from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from GrooveModel.Embedding.Embedding import DnaEmbedding
from GrooveModel.Utils.SpecialTokens import SpecialTokens


@dataclass
class SequentialDNAEmbeddingConfig:
    vocab_size: int = -1
    embedding_dim: int = -1
    normalize_embeddings: bool = False

class SequentialDNAEmbedding(DnaEmbedding):
    def __init__(self, config: SequentialDNAEmbeddingConfig):
        super(DnaEmbedding, self).__init__()
        self._embedding_dim = config.embedding_dim
        self.normalize_embedding_flag = config.normalize_embeddings
        self.vocab_size = config.vocab_size
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=SpecialTokens.PAD)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def weights(self):
        return self.embedding.weight

    def normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.normalize_embedding_flag:
            return nn.functional.layer_norm(embedding, embedding.shape[-1:])
        return embedding

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(token)
        return self.normalize_embedding(embedding)


