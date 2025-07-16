import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


def dna_collate_fn(batch):
    inputs, targets = zip(*batch)

    # Get sequence lengths BEFORE padding
    lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)

    # Sort sequences by length in descending order (required by pack_padded_sequence)
    lengths, sort_idx = lengths.sort(descending=True)
    inputs = [inputs[i] for i in sort_idx]
    targets = [targets[i] for i in sort_idx]

    # Pad sequences
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

    # Create packed sequence
    packed_inputs = pack_padded_sequence(padded_inputs, lengths, batch_first=True)

    return padded_inputs, padded_targets, packed_inputs, lengths