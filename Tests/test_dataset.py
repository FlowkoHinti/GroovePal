from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from Configs import MAX_SEQUENCE_LENGTH
from GrooveModel.CollateFunctions import pad_pack_batch, pad_truncate_batch
from GrooveModel.Datasets import DNANextTokenDataset
from GrooveModel.Tokenizer.MultiTaskDnaTokenizer import MultiTaskDnaTokenizer
from GrooveModel.Utils.SpecialTokens import SpecialTokens

# Paths
BASE_PATH = Path(__file__).resolve().parents[1]
DNA_PATH = BASE_PATH / 'Data'

# Dataset Config
ds_conf = {
    'dna_path': DNA_PATH,
    'convert_to_tensor': True,
}

PAD_VAL = float(SpecialTokens.PAD)

# --- Dataset Tests ---

def test_dataset_loads_and_returns_tensor_pair_plus_beats():
    dataset = DNANextTokenDataset(SimpleNamespace(**ds_conf), 'unit_test', tokenizer=MultiTaskDnaTokenizer)

    x, y, beats = dataset[0]

    # Inputs/targets
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == y.shape
    assert x.ndim == 2  # [seq_len, 7]
    assert x.shape[1] == 7  # DNAToken dimensions
    assert x.shape[0] >= 1  # Ensure it's not empty

    # Beat positions
    assert beats is not None
    assert isinstance(beats, torch.Tensor)
    assert beats.ndim == 1
    # tokens had length N, x is tokens[:-1], so beats length should be x_len + 1
    assert beats.shape[0] == x.shape[0] + 1


def test_dataset_multiple_samples_return_triplets():
    dataset = DNANextTokenDataset(SimpleNamespace(**ds_conf), 'unit_test', tokenizer=MultiTaskDnaTokenizer)

    sample_count = min(10, len(dataset))
    for i in range(sample_count):
        x, y, beats = dataset[i]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert isinstance(beats, torch.Tensor)
        assert x.shape == y.shape
        assert x.shape[1] == 7
        assert beats.ndim == 1
        assert beats.shape[0] == x.shape[0] + 1


# --- Collate Function Tests ---

def test_collate_fn_padding_and_alignment():
    PAD_VAL = float(SpecialTokens.PAD)
    tok = lambda v: torch.tensor([v] * 7, dtype=torch.long)

    # seq lens: 2 and 1  -> beats need to be 3 and 2 respectively
    input_1  = torch.stack([tok(1), tok(2)])     # len=2
    target_1 = torch.stack([tok(2), tok(3)])
    beats_1  = torch.tensor([10, 11, 12], dtype=torch.long)

    input_2  = torch.stack([tok(4)])             # len=1
    target_2 = torch.stack([tok(5)])
    beats_2  = torch.tensor([20, 21], dtype=torch.long)

    batch = [(input_1, target_1, beats_1), (input_2, target_2, beats_2)]
    padded_inputs, padded_targets, padded_beats, packed_inputs, lengths = pad_pack_batch(batch)

    # lengths are from inputs/targets and sorted desc
    assert torch.equal(lengths, torch.tensor([2, 1]))

    # inputs/targets are padded to max seq len (=2)
    assert torch.all(padded_inputs[1, 1:] == PAD_VAL)
    assert torch.all(padded_targets[1, 1:] == PAD_VAL)

    # beats are padded to max (inputs_len+1) (=3)
    assert padded_beats.shape[1] == padded_inputs.shape[1] + 1
    assert torch.all(padded_beats[1, 2:] == PAD_VAL)

    # alignment within the non-padded region still holds
    assert torch.equal(padded_targets[0, 0], padded_inputs[0, 1])
    assert isinstance(packed_inputs, torch.nn.utils.rnn.PackedSequence)


def test_collate_fn_with_real_data():
    bs = 16
    dataset = DNANextTokenDataset(SimpleNamespace(**ds_conf), 'unit_test', tokenizer=MultiTaskDnaTokenizer)
    loader = DataLoader(dataset, batch_size=bs, collate_fn=pad_pack_batch)

    padded_inputs, padded_targets, padded_beats, packed_inputs, lengths = next(iter(loader))

    assert isinstance(padded_inputs, torch.Tensor)
    assert isinstance(padded_targets, torch.Tensor)
    assert isinstance(padded_beats, torch.Tensor)

    assert padded_inputs.ndim == 3  # [batch, seq, 7]
    assert padded_inputs.shape == padded_targets.shape
    assert padded_inputs.shape[2] == 7

    # beats are padded at the original token length N (inputs are N-1)
    assert padded_beats.ndim == 2
    assert padded_beats.shape[0] == padded_inputs.shape[0]
    assert padded_beats.shape[1] == padded_inputs.shape[1] + 1

    assert isinstance(packed_inputs, torch.nn.utils.rnn.PackedSequence)
    assert isinstance(lengths, torch.Tensor)
    assert lengths.ndim == 1
    assert lengths.shape[0] == bs


def test_pad_truncate_batch_padding_and_truncation():
    # pad_truncate_batch returns: (padded_inputs, padded_targets, padded_beat_pos, lengths)
    PAD_VAL = float(SpecialTokens.PAD)
    seq_dim = 7

    def tok(v, n):
        return torch.stack([torch.tensor([v] * seq_dim, dtype=torch.long) for _ in range(n)])

    long_seq_len = MAX_SEQUENCE_LENGTH + 100  # exceeds max, should be truncated
    short_seq_len = MAX_SEQUENCE_LENGTH - 200  # under max, should be padded

    input_1  = tok(1, long_seq_len)
    target_1 = tok(2, long_seq_len)
    beats_1  = torch.arange(long_seq_len, dtype=torch.long)  # 0..long_seq_len-1

    input_2  = tok(3, short_seq_len)
    target_2 = tok(4, short_seq_len)
    beats_2  = torch.arange(short_seq_len, dtype=torch.long) # 0..short_seq_len-1

    batch = [(input_1, target_1, beats_1), (input_2, target_2, beats_2)]
    padded_inputs, padded_targets, padded_beats, lengths = pad_truncate_batch(batch)

    # Shapes
    assert padded_inputs.shape == (2, MAX_SEQUENCE_LENGTH, seq_dim)
    assert padded_targets.shape == (2, MAX_SEQUENCE_LENGTH, seq_dim)
    assert padded_beats.shape   == (2, MAX_SEQUENCE_LENGTH)

    # Lengths reflect truncation / true sizes (sorted desc)
    assert torch.equal(lengths, torch.tensor([MAX_SEQUENCE_LENGTH, short_seq_len]))

    # First sequence: truncated to MAX_SEQUENCE_LENGTH, no padding region to check
    # Verify content pattern instead of "no PAD values":
    #   - last (valid) beat equals lengths[0]-1 (since beats were 0..len-1)
    assert padded_beats[0, lengths[0]-1].item() == lengths[0] - 1
    #   - first few are increasing (spot check)
    assert padded_beats[0, 0].item() == 0
    assert padded_beats[0, 1].item() == 1

    # Inputs/targets for the first (truncated) sequence should contain the uniform values we set
    assert torch.all(padded_inputs[0]  == 1)
    assert torch.all(padded_targets[0] == 2)

    # Second sequence: padded past short_seq_len
    assert torch.all(padded_inputs[1, short_seq_len:]  == PAD_VAL)
    assert torch.all(padded_targets[1, short_seq_len:] == PAD_VAL)
    assert torch.all(padded_beats[1,  short_seq_len:]  == PAD_VAL)