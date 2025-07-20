import torch
import pytest
from GrooveModel.Embeddings import MultiDimDNAEmbedding, BeatPositionalEncoding


class DummyConfig:
    class embeddings:
        class instruments:
            vocab_size = 10
            embedding_dim = 8
        class velocities:
            vocab_size = 128
            embedding_dim = 4
        class offsets:
            vocab_size = 16
            embedding_dim = 6
        class time_signature:
            vocab_size = 12
            embedding_dim = 3
        class grid_factor:
            vocab_size = 10
            embedding_dim = 5
        class bpm:
            vocab_size = 256
            embedding_dim = 2

@pytest.fixture
def embedding_module():
    return MultiDimDNAEmbedding(DummyConfig())

def test_embedding_output_shape(embedding_module):
    batch_size = 4
    seq_len = 8
    num_features = 9  # full token structure
    dummy_input = torch.randint(0, 10, (batch_size, seq_len, num_features))

    output = embedding_module(dummy_input)
    expected_dim = (
        DummyConfig.embeddings.instruments.embedding_dim +
        DummyConfig.embeddings.velocities.embedding_dim +
        DummyConfig.embeddings.offsets.embedding_dim +
        DummyConfig.embeddings.time_signature.embedding_dim +
        DummyConfig.embeddings.grid_factor.embedding_dim +
        DummyConfig.embeddings.bpm.embedding_dim
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