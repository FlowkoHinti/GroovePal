from dataclasses import dataclass

import torch
from torch import nn
from typing import Sequence

from Configs import RNG_SEED
from GrooveModel.Embeddings import MultiTaskDNAEmbeddingConfig, MultiTaskDNAEmbedding
from GrooveModel.Vocab import INSTRUMENT_VOCAB_SIZE, VELOCITY_VOCAB_SIZE, BEAT_UNIT_ABSOLUTE_VOCAB_SIZE, \
    BEAT_UNIT_RELATIVE_VOCAB_SIZE, OFFSET_VOCAB_SIZE, TIME_SIGNATURE_VOCAB_SIZE, GRID_FACTOR_VOCAB_SIZE, BPM_VOCAB_SIZE
from GrooveModel.xlstm.xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
from GrooveModel.xlstm.xlstm.components.init import small_init_init_
from GrooveModel.xlstm.xlstm.utils import WeightDecayOptimGroupMixin


@dataclass
class MultiTaskDNAModelConfig(xLSTMBlockStackConfig):
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = False

class MultiTaskDNAxLSTM(WeightDecayOptimGroupMixin, nn.Module):
    config_class = MultiTaskDNAModelConfig

    def __init__(self, model_config: MultiTaskDNAModelConfig, embedding_config: MultiTaskDNAEmbeddingConfig, **kwargs):
        super().__init__()
        self.model_config = model_config

        self.xlstm_block_stack = xLSTMBlockStack(config=model_config)
        self.token_embedding = MultiTaskDNAEmbedding(embedding_config=embedding_config)
        # left out for now as i am encoding beat units as an exta feature
        # self.positional_encoding = BeatPositionalEncoding(embedding_dim=self.token_embedding.embedding_dim)
        self.emb_dropout = nn.Dropout(model_config.dropout) if model_config.add_embedding_dropout else nn.Identity()

        # TODO: MAYBE USE REGRESSION HEADS FOR VELOCITY and OFFSET
        self.multi_output_head = nn.ModuleDict({
            name: nn.Linear(self.token_embedding.embedding_dim, vocab_size, bias=False)
            for name, vocab_size in {
                'instrument': INSTRUMENT_VOCAB_SIZE,
                'velocity': VELOCITY_VOCAB_SIZE,
                'beat_unit': BEAT_UNIT_ABSOLUTE_VOCAB_SIZE if embedding_config.beat_units.absolute_beat_units else BEAT_UNIT_RELATIVE_VOCAB_SIZE,
                'offset': OFFSET_VOCAB_SIZE,
                'grid_factor': GRID_FACTOR_VOCAB_SIZE,
                'bpm': BPM_VOCAB_SIZE,
                'time_signature': TIME_SIGNATURE_VOCAB_SIZE,
            }.items()
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
        for output_head in self.multi_output_head.keys():
            small_init_init_(self.multi_output_head[output_head].weight, dim=self.token_embedding.embedding_dim)

    def forward(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.token_embedding(idx)
        x = self.emb_dropout(x)
        x = self.xlstm_block_stack(x)
        logits = {head: layer(x) for head, layer in self.multi_output_head.items()}
        return logits

    def step(
            self,
            idx: torch.Tensor,
            state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None,
            **kwargs
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, tuple[torch.Tensor, ...]]]]:
        x = self.token_embedding(idx)
        x = self.emb_dropout(x)
        x, state = self.xlstm_block_stack.step(x, state=state, **kwargs)
        logits = {head: layer(x) for head, layer in self.multi_output_head.items()}
        return logits, state

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
