from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from GrooveModel import Tokenizers
from GrooveModel.CollateFunctions import pad_pack_batch
from GrooveModel.Datasets import DNANextTokenDataset

# Paths
BASE_PATH = Path(__file__).resolve().parents[1]
DNA_PATH = BASE_PATH / 'Data' / 'dnas.json'

# Dataset Config
ds_conf = {
    'dna_path': DNA_PATH,
    'convert_to_tensor': True,
}

# --- Dataset Tests ---

def test_dataset_loads_and_returns_tensor_pair():
    dataset = DNANextTokenDataset(SimpleNamespace(**ds_conf), 'train', tokenizer=Tokenizers.MultiDimDNATokenizer)

    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == y.shape
    assert x.ndim == 2  # [seq_len, 10]
    assert x.shape[1] == 9  # DNAToken dimensions
    assert x.shape[0] >= 1  # Ensure it's not empty


def test_dataset_multiple_samples():
    dataset = DNANextTokenDataset(SimpleNamespace(**ds_conf), 'train', tokenizer=Tokenizers.MultiDimDNATokenizer)

    sample_count = min(10, len(dataset))
    for i in range(sample_count):
        x, y = dataset[i]
        assert isinstance(x, torch.Tensor)
        assert x.shape == y.shape
        assert x.shape[1] == 9


# --- Collate Function Tests ---

def test_collate_fn_padding_and_alignment():
    tok = lambda v: torch.tensor([v] * 9, dtype=torch.uint16)

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
    dataset = DNANextTokenDataset(SimpleNamespace(**ds_conf), 'train', tokenizer=Tokenizers.MultiDimDNATokenizer)
    loader = DataLoader(dataset, batch_size=bs, collate_fn=pad_pack_batch)

    batch = next(iter(loader))  # One full batch

    padded_inputs, padded_targets, packed_inputs, lengths = batch

    assert isinstance(padded_inputs, torch.Tensor)
    assert isinstance(padded_targets, torch.Tensor)
    assert padded_inputs.ndim == 3  # [batch, seq, 9]
    assert padded_inputs.shape == padded_targets.shape
    assert padded_inputs.shape[2] == 9
    assert isinstance(packed_inputs, torch.nn.utils.rnn.PackedSequence)
    assert isinstance(lengths, torch.Tensor)
    assert lengths.ndim == 1
    assert lengths.shape[0] == bs