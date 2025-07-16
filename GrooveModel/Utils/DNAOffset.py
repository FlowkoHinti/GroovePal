import numpy as np

OFFSET_TICKS_SIZE = 960

def encode_offset_ticks(offset: int, ticks_per_qn: int, start_at_zero=True,
                        resolution: int = OFFSET_TICKS_SIZE) -> int:
    """
    Convert the offset value in ticks to a quantized index.
    :param offset: The offset in ticks to convert.
    :param ticks_per_qn: Ticks per quarter note.
    :param start_at_zero: If true, shift the range to start at zero.
    :param resolution: Number of steps in the offset quantization.
    :return: The quantized offset as an integer.
    """
    # Clamp the offset to the maximum allowed range
    max_offset = ticks_per_qn
    offset = max(-max_offset, min(max_offset, offset))

    # Map offset from [-ticks_per_qn, +ticks_per_qn] to [0, resolution - 1]
    if start_at_zero:
        normalized = ((offset + ticks_per_qn) / (2 * ticks_per_qn)) * (resolution - 1)
    else:
        # Map to [-resolution//2, +resolution//2]
        normalized = (offset / ticks_per_qn) * (resolution // 2)

    return int(round(normalized))


def decode_offset_ticks(norm_offset: int, ticks_per_qn: int, start_at_zero=True,
                        norm_resolution: int = OFFSET_TICKS_SIZE) -> int:
    """
    Convert a quantized index back to the offset in ticks.
    :param norm_offset: The quantized offset index.
    :param ticks_per_qn: Ticks per quarter note.
    :param start_at_zero: Whether the index range starts at zero or is centered.
    :param norm_resolution: Number of steps in the offset quantization.
    :return: The offset in ticks.
    """
    if start_at_zero:
        # Map back from [0, resolution - 1] to [-ticks_per_qn, +ticks_per_qn]
        offset = ((norm_offset / (norm_resolution - 1)) * 2 * ticks_per_qn) - ticks_per_qn
    else:
        # Map back from [-resolution//2, +resolution//2] to [-ticks_per_qn, +ticks_per_qn]
        offset = (norm_offset / (norm_resolution // 2)) * ticks_per_qn

    return int(round(offset))