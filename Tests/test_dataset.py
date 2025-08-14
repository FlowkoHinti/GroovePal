from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from Configs import MAX_SEQUENCE_LENGTH
from GrooveModel import Tokenizers
from GrooveModel.CollateFunctions import pad_pack_batch, pad_truncate_batch
from GrooveModel.Datasets import DNANextTokenDataset
from GrooveModel.Utils.SpecialTokens import SpecialTokens

# Paths
BASE_PATH = Path(__file__).resolve().parents[1]
DNA_PATH = BASE_PATH / 'Data'

# Dataset Config
ds_conf = {
    'dna_path': DNA_PATH,
    'convert_to_tensor': True,
}

# --- Dataset Tests ---

def test_dataset_loads_and_returns_tensor_pair():
    dataset = DNANextTokenDataset(SimpleNamespace(**ds_conf), 'unit_test', tokenizer=Tokenizers.MultiTaskDnaTokenizer)

    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == y.shape
    assert x.ndim == 2  # [seq_len, 10]
    assert x.shape[1] == 7  # DNAToken dimensions
    assert x.shape[0] >= 1  # Ensure it's not empty


def test_dataset_multiple_samples():
    dataset = DNANextTokenDataset(SimpleNamespace(**ds_conf), 'unit_test', tokenizer=Tokenizers.MultiTaskDnaTokenizer)

    sample_count = min(10, len(dataset))
    for i in range(sample_count):
        x, y = dataset[i]
        assert isinstance(x, torch.Tensor)
        assert x.shape == y.shape
        assert x.shape[1] == 7


# --- Collate Function Tests ---

def test_collate_fn_padding_and_alignment():
    tok = lambda v: torch.tensor([v] * 7, dtype=torch.int)

    input_1 = torch.stack([tok(1), tok(2)])
    target_1 = torch.stack([tok(2), tok(3)])
    input_2 = torch.stack([tok(4)])
    target_2 = torch.stack([tok(5)])

    batch = [(input_1, target_1), (input_2, target_2)]
    padded_inputs, padded_targets, packed_inputs, lengths = pad_pack_batch(batch)

    # Check lengths
    assert torch.equal(lengths, torch.tensor([2, 1]))

    # Check padding
    assert torch.all(padded_inputs[1, 1:] == 0)
    assert torch.all(padded_targets[1, 1:] == 0)

    # Check alignment
    assert torch.equal(padded_targets[0, 0], padded_inputs[0, 1])

def test_collate_fn_with_real_data():
    bs = 16
    dataset = DNANextTokenDataset(SimpleNamespace(**ds_conf), 'unit_test', tokenizer=Tokenizers.MultiTaskDnaTokenizer)
    loader = DataLoader(dataset, batch_size=bs, collate_fn=pad_pack_batch)

    batch = next(iter(loader))  # One full batch

    padded_inputs, padded_targets, packed_inputs, lengths = batch

    assert isinstance(padded_inputs, torch.Tensor)
    assert isinstance(padded_targets, torch.Tensor)
    assert padded_inputs.ndim == 3  # [batch, seq, 7]
    assert padded_inputs.shape == padded_targets.shape
    assert padded_inputs.shape[2] == 7
    assert isinstance(packed_inputs, torch.nn.utils.rnn.PackedSequence)
    assert isinstance(lengths, torch.Tensor)
    assert lengths.ndim == 1
    assert lengths.shape[0] == bs



def test_pad_truncate_batch_padding_and_truncation():
    seq_dim = 7
    tok = lambda v, n: torch.stack([torch.tensor([v] * seq_dim, dtype=torch.int) for _ in range(n)])

    long_seq_len = MAX_SEQUENCE_LENGTH + 100  # exceeds max, should be truncated
    short_seq_len = MAX_SEQUENCE_LENGTH - 200  # under max, should be padded

    input_1 = tok(1, long_seq_len)     # [1124, 7]
    target_1 = tok(2, long_seq_len)
    input_2 = tok(3, short_seq_len)    # [824, 7]
    target_2 = tok(4, short_seq_len)

    batch = [(input_1, target_1), (input_2, target_2)]
    padded_inputs, padded_targets, lengths = pad_truncate_batch(batch)

    # Check output shapes
    assert padded_inputs.shape == (2, MAX_SEQUENCE_LENGTH, seq_dim)
    assert padded_targets.shape == (2, MAX_SEQUENCE_LENGTH, seq_dim)

    # Check lengths reflect truncation and true sequence size
    assert torch.equal(lengths, torch.tensor([MAX_SEQUENCE_LENGTH, short_seq_len]))

    # First sequence should be truncated
    assert torch.all(padded_inputs[0] == 1)
    assert torch.all(padded_targets[0] == 2)

    # Second sequence should be padded
    pad_value = float(SpecialTokens.PAD)
    assert torch.all(padded_inputs[1, short_seq_len:] == pad_value)
    assert torch.all(padded_targets[1, short_seq_len:] == pad_value)