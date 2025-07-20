from GrooveModel.Utils.SpecialTokens import SPECIAL_TOKEN_SIZE

OFFSET_TICKS_RESOLUTION = 960  # usable steps excluding special tokens (+-480 ticks)
OFFSET_TOKEN_SIZE = OFFSET_TICKS_RESOLUTION + SPECIAL_TOKEN_SIZE  # total size including special tokens


def encode_offset_ticks(offset: int, ticks_per_qn: int, start_at_zero=True,
                        resolution: int = OFFSET_TICKS_RESOLUTION) -> int:
    """
    Convert the offset value in ticks to a quantized index, offset by SPECIAL_TOKEN_SIZE.
    :param offset: The offset in ticks to convert.
    :param ticks_per_qn: Ticks per quarter note.
    :param start_at_zero: If true, shift the range to start at zero.
    :param resolution: Number of steps in the offset quantization (excluding special tokens).
    :return: The quantized offset as an integer (including SPECIAL_TOKEN_SIZE shift).
    """
    max_offset = ticks_per_qn
    offset = max(-max_offset, min(max_offset, offset))

    if start_at_zero:
        normalized = ((offset + ticks_per_qn) / (2 * ticks_per_qn)) * (resolution - 1)
    else:
        normalized = (offset / ticks_per_qn) * (resolution // 2)

    quantized_index = int(round(normalized))
    return quantized_index + SPECIAL_TOKEN_SIZE


def decode_offset_ticks(norm_offset: int, ticks_per_qn: int, start_at_zero=True,
                        resolution: int = OFFSET_TICKS_RESOLUTION) -> int:
    """
    Convert a quantized index (with special token offset) back to the offset in ticks.
    :param norm_offset: The quantized offset index including SPECIAL_TOKEN_SIZE offset.
    :param ticks_per_qn: Ticks per quarter note.
    :param start_at_zero: Whether the index range starts at zero or is centered.
    :param resolution: Number of quantization steps used (excluding special tokens).
    :return: The decoded offset in ticks.
    """
    quantized_index = norm_offset - SPECIAL_TOKEN_SIZE

    if start_at_zero:
        offset = ((quantized_index / (resolution - 1)) * 2 * ticks_per_qn) - ticks_per_qn
    else:
        offset = (quantized_index / (resolution // 2)) * ticks_per_qn

    return int(round(offset))