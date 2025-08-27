import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from Configs import MAX_SEQUENCE_LENGTH
from GrooveModel.Utils.SpecialTokens import SpecialTokens


def _prepare_batch(
        batch,
        *,
        max_len: int | None = None,
        do_pack: bool = False,
        batch_first: bool = True,
) -> tuple:
    """
    Common batch prep: optional truncate -> length -> sort(desc) -> pad -> (optional) pack.

    Returns (padded_inputs, padded_targets, lengths) unless do_pack=True,
    in which case it returns (padded_inputs, padded_targets, packed_inputs, lengths).

    Notes:
    - `lengths` are sorted (desc) to match the returned padded tensors.
    - `pack_padded_sequence` requires sorted lengths; we enforce that here.
    """
    inputs, targets, beat_pos = zip(*batch)  # sequences of 1D/2D tensors

    # Optional truncation
    if max_len is not None:
        inputs = [seq[:max_len] for seq in inputs]
        targets = [seq[:max_len] for seq in targets]
        beat_pos = [seq[:max_len] for seq in beat_pos]

    # Lengths BEFORE padding
    lengths = torch.as_tensor([len(seq) for seq in inputs], dtype=torch.long)

    # Sort by length (desc) for pack_padded_sequence
    lengths, sort_idx = lengths.sort(descending=True)
    inputs = [inputs[i] for i in sort_idx]
    targets = [targets[i] for i in sort_idx]
    beat_pos = [beat_pos[i] for i in sort_idx]

    # Pad
    pad_val = float(SpecialTokens.PAD)
    padded_inputs = pad_sequence(inputs, batch_first=batch_first, padding_value=pad_val)
    padded_targets = pad_sequence(targets, batch_first=batch_first, padding_value=pad_val)
    padded_beat_pos = pad_sequence(beat_pos, batch_first=batch_first, padding_value=pad_val)

    if do_pack:
        # pack_padded_sequence expects CPU lengths
        packed_inputs = pack_padded_sequence(
            padded_inputs, lengths.cpu(), batch_first=batch_first, enforce_sorted=True
        )
        return padded_inputs, padded_targets, padded_beat_pos, packed_inputs, lengths

    return padded_inputs, padded_targets, padded_beat_pos, lengths


def pad_pack_batch(batch):
    """Pad + pack (no truncation)."""
    return _prepare_batch(batch, do_pack=True)


def pad_batch(batch):
    """Pad only (no truncation, no packing)."""
    return _prepare_batch(batch, do_pack=False)


def pad_truncate_batch(batch):
    """Truncate to MAX_SEQUENCE_LENGTH, then pad (no packing)."""
    return _prepare_batch(batch, max_len=MAX_SEQUENCE_LENGTH, do_pack=False)
