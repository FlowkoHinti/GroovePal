import torch

from GrooveModel.Utils.SpecialTokens import SPECIAL_TOKEN_SIZE

OFFSET_TICKS_RESOLUTION = 120  # usable steps excluding special tokens
OFFSET_TOKEN_SIZE = OFFSET_TICKS_RESOLUTION + SPECIAL_TOKEN_SIZE  # total size including special tokens


def encode_offset_ticks(offset: int, ticks_per_grid_unit: int = OFFSET_TICKS_RESOLUTION, start_at_zero=False,
                        resolution: int = OFFSET_TICKS_RESOLUTION, include_padding: bool=True) -> int:
    """
    Convert the offset value in ticks to a quantized index, offset by SPECIAL_TOKEN_SIZE.
    Offsets are clamped to [-ticks_per_grid_unit/2, +ticks_per_grid_unit/2].
    """
    max_offset = ticks_per_grid_unit // 2
    offset = max(-max_offset, min(max_offset, offset))

    if start_at_zero:
        normalized = ((offset + max_offset) / (2 * max_offset)) * (resolution - 1)
    else:
        normalized = (offset / max_offset) * (resolution // 2)

    quantized_index = int(round(normalized))
    return quantized_index + SPECIAL_TOKEN_SIZE if include_padding else quantized_index


def decode_offset_ticks(norm_offset: int, ticks_per_grid_unit: int = OFFSET_TICKS_RESOLUTION, start_at_zero=False,
                        resolution: int = OFFSET_TICKS_RESOLUTION, include_padding: bool=True) -> int:
    """
    Convert a quantized index (with special token offset) back to the offset in ticks.
    Assumes offsets are within [-ticks_per_grid_unit/2, +ticks_per_grid_unit/2].
    """
    max_offset = ticks_per_grid_unit // 2
    quantized_index = norm_offset - SPECIAL_TOKEN_SIZE if include_padding else norm_offset

    if start_at_zero:
        offset = ((quantized_index / (resolution - 1)) * (2 * max_offset)) - max_offset
    else:
        offset = (quantized_index / (resolution // 2)) * max_offset

    return int(round(offset))


def normalize_offset(offset: int, ticks_per_grid_unit: int = OFFSET_TICKS_RESOLUTION, start_at_zero: bool = False) -> float:
    """
    Normalize an offset in ticks to a float in [-1, 1] or [0, 1].
    Offsets are clamped to [-ticks_per_grid_unit/2, +ticks_per_grid_unit/2].
    """
    max_offset = ticks_per_grid_unit // 2
    offset = max(-max_offset, min(max_offset, offset))

    if start_at_zero:
        normalized = (offset + max_offset) / (2 * max_offset)
    else:
        normalized = offset / max_offset

    return float(normalized)


def denormalize_offset(norm_offset: float, ticks_per_grid_unit: int = OFFSET_TICKS_RESOLUTION, start_at_zero: bool = False) -> int:
    """
    Convert a normalized offset back to ticks.
    Produces values in [-ticks_per_grid_unit/2, +ticks_per_grid_unit/2].
    """
    max_offset = ticks_per_grid_unit // 2

    if start_at_zero:
        offset = (norm_offset * (2 * max_offset)) - max_offset
    else:
        offset = norm_offset * max_offset

    return int(round(offset))

def normalize_offset_tensor(off_ids, dtype=torch.float32) -> torch.Tensor:
    off_idx = (off_ids - SPECIAL_TOKEN_SIZE).clamp(min=0, max=OFFSET_TICKS_RESOLUTION - 1)
    off_0_1 = off_idx.to(dtype) / float(max(1, OFFSET_TICKS_RESOLUTION - 1))
    return 2.0 * off_0_1 - 1.0
