import pytest
import torch
from dacite import Config as DaciteConfig
from dacite import from_dict
from omegaconf import OmegaConf

from GrooveModel.Embedding.BeatPositionalEncoding import BeatPositionalEncoding
from GrooveModel.Embedding.MultiTaskDnaEmbedding import MultiTaskDNAEmbeddingConfig, MultiTaskDNAEmbedding

embedding_config_string = f"""
instruments:
    embedding_dim: 8
velocities:
    embedding_dim: 4
offsets:
    embedding_dim: 4
time_signature:
    embedding_dim: 4
grid_factor:
    embedding_dim: 8
bpm:
    embedding_dim: 6
beat_units:
    embedding_dim: 8
    absolute_beat_units: false
normalize_embeddings: false
"""

embedding_config = OmegaConf.create(embedding_config_string)

embedding_config = from_dict(
    MultiTaskDNAEmbeddingConfig,
    OmegaConf.to_container(embedding_config, resolve=True),
    config=DaciteConfig(strict=True)
)


@pytest.fixture
def embedding_module():
    return MultiTaskDNAEmbedding(embedding_config)


def test_embedding_output_shape(embedding_module):
    batch_size = 4
    seq_len = 8
    num_features = 7  # full token structure
    dummy_input = torch.cat([
        torch.randint(1, 8, (batch_size, seq_len, 1)),  # instrument
        torch.randint(1, 129, (batch_size, seq_len, 1)),  # velocity
        torch.randint(1, 73, (batch_size, seq_len, 1)),  # beat_unit
        torch.randint(1, 121, (batch_size, seq_len, 1)),  # offset
        torch.randint(1, 5, (batch_size, seq_len, 1)),  # grid
        torch.randint(1, 40, (batch_size, seq_len, 1)),  # bpm
        torch.randint(1, 11, (batch_size, seq_len, 1)),  # time_signature
    ], dim=2)

    output = embedding_module(dummy_input)
    expected_dim = (
            embedding_module.sub_embeddings['instrument'].embedding_dim +
            embedding_module.sub_embeddings['velocity'].embedding_dim +
            embedding_module.sub_embeddings['beat_unit'].embedding_dim +
            embedding_module.sub_embeddings['offset'].embedding_dim +
            embedding_module.sub_embeddings['time_signature'].embedding_dim +
            embedding_module.sub_embeddings['grid_factor'].embedding_dim +
            embedding_module.sub_embeddings['bpm'].embedding_dim
    )

    assert output.shape == (batch_size, seq_len, expected_dim)


@pytest.fixture
def pos_encoder():
    return BeatPositionalEncoding(embedding_dim=32, max_len=100)


def test_positional_encoding_adds_embedding(pos_encoder):
    batch_size = 2
    seq_len = 5
    d_model = 32
    dummy_input = torch.zeros((batch_size, seq_len, d_model))
    beat_units = torch.tensor([
        [0, 1, 2, 3, 4],
        [4, 3, 2, 1, 0]
    ])

    output = pos_encoder(dummy_input, beat_unit=beat_units)

    assert output.shape == (batch_size, seq_len, d_model)
    # Should not be all zeros since we added encoding
    assert not torch.allclose(output, dummy_input)


def test_default_position_fallback(pos_encoder):
    dummy_input = torch.zeros((1, 4, 32))  # no beat_unit provided
    output = pos_encoder(dummy_input)
    assert output.shape == (1, 4, 32)
