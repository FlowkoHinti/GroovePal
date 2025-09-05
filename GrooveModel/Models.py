from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from GrooveModel.Embedding.BeatPositionalEncoding import BeatPositionalEncoding
from GrooveModel.Embedding.MultiTaskDnaEmbedding import MultiTaskDNAEmbedding, MultiTaskDNAEmbeddingConfig
from GrooveModel.Embedding.SequentialDnaEmbedding import SequentialDNAEmbeddingConfig, SequentialDNAEmbedding
from GrooveModel.Utils.BeatUnit import MAX_GRID_UNITS_PER_BAR, MAX_GRID_UNITS_PER_SONG
from GrooveModel.Vocab import INSTRUMENT_VOCAB_SIZE, BEAT_UNIT_ABSOLUTE_VOCAB_SIZE, \
    BEAT_UNIT_RELATIVE_VOCAB_SIZE, TIME_SIGNATURE_VOCAB_SIZE, GRID_FACTOR_VOCAB_SIZE, BPM_VOCAB_SIZE
from GrooveModel.xlstm.xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
from GrooveModel.xlstm.xlstm.components.init import small_init_init_
from GrooveModel.xlstm.xlstm.utils import WeightDecayOptimGroupMixin


@dataclass
class ModelConfigxLstm(xLSTMBlockStackConfig):
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = False
    positional_encoding: bool = False


class MultiTaskDNAxLSTM(WeightDecayOptimGroupMixin, nn.Module):
    config_class = ModelConfigxLstm

    def __init__(self, model_config: ModelConfigxLstm, embedding_config: MultiTaskDNAEmbeddingConfig, **kwargs):
        super().__init__()
        self.model_config = model_config

        self.xlstm_block_stack = xLSTMBlockStack(config=model_config)
        self.token_embedding = MultiTaskDNAEmbedding(embedding_config=embedding_config)
        # left out for now as i am encoding beat units as an exta feature
        # self.positional_encoding = BeatPositionalEncoding(embedding_dim=self.token_embedding.embedding_dim)
        self.emb_dropout = nn.Dropout(model_config.dropout) if model_config.add_embedding_dropout else nn.Identity()

        self.classification_heads = nn.ModuleDict({
            name: nn.Linear(self.token_embedding.embedding_dim, vocab_size)
            for name, vocab_size in {
                'instrument': INSTRUMENT_VOCAB_SIZE,
                'beat_unit': BEAT_UNIT_ABSOLUTE_VOCAB_SIZE if embedding_config.beat_units.absolute_beat_units else BEAT_UNIT_RELATIVE_VOCAB_SIZE,
                'grid_factor': GRID_FACTOR_VOCAB_SIZE,
                'bpm': BPM_VOCAB_SIZE,
                'time_signature': TIME_SIGNATURE_VOCAB_SIZE,
            }.items()
        })

        self.regression_heads = nn.ModuleDict({
            'velocity': nn.Sequential(nn.Linear(self.token_embedding.embedding_dim, 1),
                                      nn.Sigmoid()),
            'offset': nn.Sequential(nn.Linear(self.token_embedding.embedding_dim, 1),
                                    nn.Tanh())
        })
        # TODO: Think about implementing tie weights (model only learns 1 representation):
        # same weights for input and output: https://arxiv.org/abs/1608.05859
        # -> maybe take the instrument embedding dims (first x dims of total embedding) and tie them to the first hidden_state dims
        # probably will not work as all the other dims also affect value
        # -> utilize the same amount of dims as embedding dims for all sub parts (eg. instrument 8 emb dim, first 8 dims of hidden state)
        # CAVEAT: lose extra information from all other dims -> model either better or worse (param efficiency vs more complex modelling)
        # if model_config.tie_weights:
        #     self.lm_head.weight = self.token_embedding.weight

    def reset_parameters(self):
        # Reset xLSTM blocks
        self.xlstm_block_stack.reset_parameters()

        # Initialize token embedding weights according to 'Transformers without Tears'
        for embedding in self.token_embedding.sub_embeddings.values():
            small_init_init_(embedding.weight, dim=embedding.embedding_dim)

        # Initialize head weights
        for output_head in self.classification_heads.keys():
            small_init_init_(self.classification_heads[output_head].weight, dim=self.token_embedding.embedding_dim)
        for output_head in self.regression_heads.keys():
            small_init_init_(self.regression_heads[output_head][0].weight, dim=self.token_embedding.embedding_dim)

    def forward(
            self,
                idx: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        x, beat_pos = idx
        x = self.token_embedding(x)
        x = self.emb_dropout(x)
        x = self.xlstm_block_stack(x)
        logits = {head: layer(x) for head, layer in self.classification_heads.items()}
        reg_outputs = {head: layer(x) for head, layer in self.regression_heads.items()}

        return logits, reg_outputs

    def step(
            self,
            idx: tuple[torch.Tensor, torch.Tensor],
            state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None,
            **kwargs
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, dict[str, tuple[torch.Tensor, ...]]]]:
        x, beat_pos = idx
        x = self.token_embedding(x)
        x = self.emb_dropout(x)
        x, state = self.xlstm_block_stack.step(x, state=state, **kwargs)
        logits = {head: layer(x) for head, layer in self.classification_heads.items()}
        reg_outputs = {head: layer(x) for head, layer in self.regression_heads.items()}

        return logits, reg_outputs, state

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        weight_decay, no_weight_decay = super()._create_weight_decay_optim_groups(**kwargs)

        # Convert to lists so we can modify them
        weight_decay = list(weight_decay)
        no_weight_decay = list(no_weight_decay)

        for name, embedding in self.token_embedding.sub_embeddings.items():
            # Remove the embedding weight from weight_decay if it exists
            weight_decay = [p for p in weight_decay if p is not embedding.weight]

            # Add to the correct group
            if self.model_config.weight_decay_on_embedding:
                weight_decay.append(embedding.weight)
            else:
                no_weight_decay.append(embedding.weight)

        return tuple(weight_decay), tuple(no_weight_decay)


class SequentialDNAxLSTM(WeightDecayOptimGroupMixin, nn.Module):
    config_class = ModelConfigxLstm

    def __init__(self,
                 model_config: ModelConfigxLstm,
                 embedding_config: SequentialDNAEmbeddingConfig,
                 absolute_grid_units: bool = False, **kwargs):
        super().__init__()
        self.model_config = model_config

        self.xlstm_block_stack = xLSTMBlockStack(config=model_config)
        self.token_embedding = SequentialDNAEmbedding(config=embedding_config)

        self.bpe = model_config.positional_encoding
        if self.bpe:
            self.positional_encoding = BeatPositionalEncoding(embedding_dim=self.token_embedding.embedding_dim,
                                                              max_len=MAX_GRID_UNITS_PER_SONG if absolute_grid_units else MAX_GRID_UNITS_PER_BAR)
        self.emb_dropout = nn.Dropout(model_config.dropout) if model_config.add_embedding_dropout else nn.Identity()

        self.output_head = nn.Linear(
            in_features=embedding_config.embedding_dim,
            out_features=embedding_config.vocab_size,
            bias=False)

        if model_config.tie_weights:
            self.output_head.weight = self.token_embedding.weights()

    def reset_parameters(self):
        self.xlstm_block_stack.reset_parameters()

        small_init_init_(self.token_embedding.weights(), dim=self.model_config.embedding_dim)

        if not self.model_config.tie_weights:
            small_init_init_(self.output_head.weight, dim=self.model_config.embedding_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x, beat_pos = idx
        x = self.token_embedding(x)
        if self.bpe:
            x = self.positional_encoding(x, beat_pos)
        x = self.emb_dropout(x)
        x = self.xlstm_block_stack(x)
        logits = self.output_head(x)
        return logits

    def step(
            self, idx: tuple[torch.Tensor, torch.Tensor], state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None, **kwargs
    ) -> tuple[torch.Tensor, dict[str, dict[str, tuple[torch.Tensor, ...]]]]:
        x, beat_pos = idx
        x = self.token_embedding(x)
        if self.bpe:
            x = self.positional_encoding(x, beat_pos)
        x = self.emb_dropout(x)
        x, state = self.xlstm_block_stack.step(x, state=state, **kwargs)
        logits = self.output_head(x)
        return logits, state

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        weight_decay, no_weight_decay = super()._create_weight_decay_optim_groups(**kwargs)
        # remove token embedding and add it to the correct group, according to the config
        weight_decay = list(weight_decay)
        removed = 0
        for idx in range(len(weight_decay)):
            if weight_decay[idx - removed] is self.token_embedding.weights():
                weight_decay.pop(idx - removed)
                removed += 1
        weight_decay = tuple(weight_decay)
        if self.model_config.weight_decay_on_embedding:
            weight_decay += (self.token_embedding.weights(),)
        else:
            no_weight_decay += (self.token_embedding.weights(),)

        return weight_decay, no_weight_decay
