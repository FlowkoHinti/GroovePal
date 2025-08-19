from abc import abstractmethod, ABC

import torch
from torch import nn


class DnaEmbedding(nn.Module, ABC):
    def __init__(self):
        super(DnaEmbedding, self).__init__()

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @abstractmethod
    def normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, token: torch.Tensor) -> torch.Tensor:
        pass
