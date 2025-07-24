import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from Configs import MAX_SEQUENCE_LENGTH
from GrooveModel.Utils.SpecialTokens import SpecialTokens

def pad_pack_batch(batch):
    inputs, targets = zip(*batch)

    # Get sequence lengths BEFORE padding
    lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)

    # Sort sequences by length in descending order (required by pack_padded_sequence)
    lengths, sort_idx = lengths.sort(descending=True)
    inputs = [inputs[i] for i in sort_idx]
    targets = [targets[i] for i in sort_idx]

    # Pad sequences
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=float(SpecialTokens.PAD))
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=float(SpecialTokens.PAD))

    # Create packed sequence
    packed_inputs = pack_padded_sequence(padded_inputs, lengths, batch_first=True)

    return padded_inputs, padded_targets, packed_inputs, lengths


def pad_batch(batch):
    inputs, targets = zip(*batch)

    # Get sequence lengths BEFORE padding
    lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)

    # Sort sequences by length in descending order
    sorted_lengths, sorted_idx = lengths.sort(descending=True)
    inputs = [inputs[i] for i in sorted_idx]
    targets = [targets[i] for i in sorted_idx]

    # Pad sequences
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=float(SpecialTokens.PAD))
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=float(SpecialTokens.PAD))

    return padded_inputs, padded_targets, lengths

def pad_truncate_batch(batch):
    inputs, targets = zip(*batch)

    # Truncate each sequence to max_length
    truncated_inputs = [seq[:MAX_SEQUENCE_LENGTH] for seq in inputs]
    truncated_targets = [seq[:MAX_SEQUENCE_LENGTH] for seq in targets]

    # Get sequence lengths AFTER truncation, BEFORE padding
    lengths = torch.tensor([min(len(seq), MAX_SEQUENCE_LENGTH) for seq in truncated_inputs], dtype=torch.long)

    # Sort by sequence lengths (descending)
    sorted_lengths, sorted_idx = lengths.sort(descending=True)
    truncated_inputs = [truncated_inputs[i] for i in sorted_idx]
    truncated_targets = [truncated_targets[i] for i in sorted_idx]

    # Pad sequences
    padded_inputs = pad_sequence(truncated_inputs, batch_first=True, padding_value=float(SpecialTokens.PAD))
    padded_targets = pad_sequence(truncated_targets, batch_first=True, padding_value=float(SpecialTokens.PAD))

    return padded_inputs, padded_targets, sorted_lengths