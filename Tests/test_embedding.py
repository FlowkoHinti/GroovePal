import sys

import pytest
import torch
from dacite import Config as DaciteConfig
from dacite import from_dict
from omegaconf import OmegaConf

from GrooveModel.Embedding.BeatPositionalEncoding import BeatPositionalEncoding
from GrooveModel.Embedding.MultiTaskDnaEmbedding import (
    MultiTaskDNAEmbeddingConfig,
    MultiTaskDNAEmbedding,
)
from GrooveModel.Embedding.SequentialDnaEmbedding import SequentialDNAEmbedding, SequentialDNAEmbeddingConfig
from GrooveModel.Utils.BeatsPerMinute import NUM_BPM_BINS
from GrooveModel.Utils.DNAOffset import OFFSET_TICKS_RESOLUTION
from GrooveModel.Utils.DNAVelocity import VELOCITY_MAX, EFFECTIVE_VELOCITY_RESOLUTION
from GrooveModel.Utils.SpecialTokens import SpecialTokens


# ----------------------------
# config for MultiTask
# ----------------------------
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

_embedding_config = OmegaConf.create(embedding_config_string)

embedding_config = from_dict(
    MultiTaskDNAEmbeddingConfig,
    OmegaConf.to_container(_embedding_config, resolve=True),
    config=DaciteConfig(strict=True),
)


# ----------------------------
# MultiTaskDNAEmbedding tests
# ----------------------------
@pytest.fixture
def embedding_module():
    return MultiTaskDNAEmbedding(embedding_config)


class TestMultiTaskDNAEmbedding:
    def test_embedding_output_shape(self, embedding_module):
        batch_size = 4
        seq_len = 8
        # full token structure (instrument, velocity, beat_unit, offset, grid, bpm, time_signature)
        num_features = 7
        assert num_features == 7  # sanity

        dummy_input = torch.cat(
            [
                torch.randint(1, 8, (batch_size, seq_len, 1)),  # instrument
                torch.randint(1, EFFECTIVE_VELOCITY_RESOLUTION + 1, (batch_size, seq_len, 1)),  # velocity
                torch.randint(1, 73, (batch_size, seq_len, 1)),  # beat_unit
                torch.randint(1, OFFSET_TICKS_RESOLUTION, (batch_size, seq_len, 1)),  # offset
                torch.randint(1, 5, (batch_size, seq_len, 1)),  # grid
                torch.randint(1, NUM_BPM_BINS, (batch_size, seq_len, 1)),  # bpm
                torch.randint(1, 11, (batch_size, seq_len, 1)),  # time_signature
            ],
            dim=2,
        )

        output = embedding_module(dummy_input)
        expected_dim = (
            embedding_module.sub_embeddings["instrument"].embedding_dim
            + embedding_module.sub_embeddings["velocity"].embedding_dim
            + embedding_module.sub_embeddings["beat_unit"].embedding_dim
            + embedding_module.sub_embeddings["offset"].embedding_dim
            + embedding_module.sub_embeddings["time_signature"].embedding_dim
            + embedding_module.sub_embeddings["grid_factor"].embedding_dim
            + embedding_module.sub_embeddings["bpm"].embedding_dim
        )

        assert output.shape == (batch_size, seq_len, expected_dim)


# ----------------------------
# BeatPositionalEncoding tests
# ----------------------------
@pytest.fixture
def pos_encoder():
    return BeatPositionalEncoding(embedding_dim=32, max_len=100)


def test_positional_encoding_adds_embedding(pos_encoder):
    batch_size = 2
    seq_len = 5
    d_model = 32
    dummy_input = torch.zeros((batch_size, seq_len, d_model))
    beat_units = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])

    output = pos_encoder(dummy_input, beat_unit=beat_units)

    assert output.shape == (batch_size, seq_len, d_model)
    # Should not be all zeros since we added encoding
    assert not torch.allclose(output, dummy_input)


def test_default_position_fallback(pos_encoder):
    dummy_input = torch.zeros((1, 4, 32))  # no beat_unit provided
    output = pos_encoder(dummy_input)
    assert output.shape == (1, 4, 32)


# ----------------------------
# SequentialDnaEmbedding tests
# ----------------------------
# ----------------------------
# SequentialDnaEmbedding tests (config-based)
# ----------------------------
@pytest.fixture(params=[False, True], ids=["no_norm", "with_norm"])
def sequential_embedding(request):
    torch.manual_seed(0)
    cfg = SequentialDNAEmbeddingConfig(
        vocab_size=101,
        embedding_dim=16,
        normalize_embeddings=request.param,
    )
    return SequentialDNAEmbedding(cfg)


class TestSequentialDnaEmbedding:
    def test_embedding_dim_property(self, sequential_embedding):
        assert sequential_embedding.embedding_dim == sequential_embedding._embedding_dim

    def test_output_shape(self, sequential_embedding):
        batch_size = 3
        seq_len = 7
        tokens = torch.randint(
            low=1, high=sequential_embedding.vocab_size, size=(batch_size, seq_len)
        )
        out = sequential_embedding(tokens)
        assert out.shape == (batch_size, seq_len, sequential_embedding.embedding_dim)

    def test_padding_yields_zero_vector(self, sequential_embedding):
        batch_size = 2
        seq_len = 5
        tokens = torch.randint(
            low=1, high=sequential_embedding.vocab_size, size=(batch_size, seq_len)
        )
        tokens[0, 0] = SpecialTokens.PAD
        tokens[1, 3] = SpecialTokens.PAD

        out = sequential_embedding(tokens)

        assert torch.allclose(out[0, 0], torch.zeros_like(out[0, 0]))
        assert torch.allclose(out[1, 3], torch.zeros_like(out[1, 3]))

    def test_layer_norm_behavior_matches_config(self, sequential_embedding):
        """No skips: assert the correct behavior for both normalization settings."""
        batch_size = 2
        seq_len = 6
        tokens = torch.randint(1, sequential_embedding.vocab_size, (batch_size, seq_len))

        raw = sequential_embedding.embedding(tokens)   # raw lookup
        out = sequential_embedding(tokens)             # forward()

        if sequential_embedding.normalize_embedding_flag:
            # With LayerNorm: per (B,T) position, last-dim mean≈0 and std≈1
            means = out.mean(dim=-1)
            stds = out.std(dim=-1, unbiased=False)
            assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
            assert torch.allclose(stds, torch.ones_like(stds), atol=1e-4)
            # Should differ from raw embedding (avoid degenerate equality)
            assert not torch.allclose(out, raw)
        else:
            # No normalization: forward returns raw embedding exactly
            assert torch.allclose(out, raw)

    def test_layer_norm_applied_when_enabled(self, sequential_embedding):
        # Only assert the LayerNorm property when enabled
        if not sequential_embedding.normalize_embedding_flag:
            pytest.skip("This check is only meaningful with normalization enabled")

        batch_size = 2
        seq_len = 6
        tokens = torch.randint(1, sequential_embedding.vocab_size, (batch_size, seq_len))
        out = sequential_embedding(tokens)  # [B, T, D]

        means = out.mean(dim=-1)  # [B, T]
        stds = out.std(dim=-1, unbiased=False)  # [B, T]
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
        assert torch.allclose(stds, torch.ones_like(stds), atol=1e-4)

    def test_no_norm_returns_raw_embedding(self, sequential_embedding):
        # This is the companion check for the no-norm param
        if sequential_embedding.normalize_embedding_flag:
            pytest.skip("This check is only meaningful with normalization disabled")

        batch_size = 2
        seq_len = 6
        tokens = torch.randint(1, sequential_embedding.vocab_size, (batch_size, seq_len))
        raw = sequential_embedding.embedding(tokens)  # raw lookup
        out = sequential_embedding(tokens)  # forward()
        # With normalize disabled, forward should be the raw embedding exactly
        assert torch.allclose(out, raw)