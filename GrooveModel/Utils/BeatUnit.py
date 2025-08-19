from Configs import MAX_SEQUENCE_LENGTH
from GrooveModel.Utils.DNAGridFactor import GridFactors
from GrooveModel.Utils.SpecialTokens import SPECIAL_TOKEN_SIZE
from GrooveModel.Utils.TimeSignatures import decode_time_signature, RemappedTimeSignatures

MAX_GRID_UNITS_PER_BAR = GridFactors.SixteenthTriplet_Grid * decode_time_signature(RemappedTimeSignatures.Time_12_8)[0]
MAX_GRID_UNITS_PER_SONG = MAX_SEQUENCE_LENGTH

# Token vocab sizes
BEAT_UNIT_TOKEN_SIZE_RELATIVE = MAX_GRID_UNITS_PER_BAR + SPECIAL_TOKEN_SIZE
BEAT_UNIT_TOKEN_SIZE_ABSOLUTE = MAX_GRID_UNITS_PER_SONG + SPECIAL_TOKEN_SIZE


def encode_beat_unit(position: int, absolute: bool = False, include_padding: bool = True) -> int:
    """
    Encodes a beat unit (grid position) into a token ID with special token offset.

    :param position: Position in grid units (absolute or relative to bar).
    :param absolute: Whether the position is absolute (entire song) or relative (per bar).
    :param include_padding: Whether to include padding token.
    :return: Token ID (with SPECIAL_TOKEN_SIZE offset).
    """
    max_val = MAX_GRID_UNITS_PER_SONG if absolute else MAX_GRID_UNITS_PER_BAR

    if not (0 <= position < max_val):
        raise ValueError(f"Beat unit position out of bounds: {position} (max allowed: {max_val - 1})")

    return position + SPECIAL_TOKEN_SIZE if include_padding else position


def decode_beat_unit(token: int, absolute: bool = False, include_padding: bool = True) -> int:
    """
    Decodes a token ID back to a grid unit position.

    :param token: Token ID with SPECIAL_TOKEN_SIZE offset.
    :param absolute: Whether the original value was absolute or relative.
    :param include_padding: Whether to include padding token.
    :return: Position in grid units.
    """
    position = token - SPECIAL_TOKEN_SIZE if include_padding else token
    max_val = MAX_GRID_UNITS_PER_SONG if absolute else MAX_GRID_UNITS_PER_BAR

    if not (0 <= position < max_val):
        raise ValueError(f"Decoded beat unit position out of bounds: {position} (max allowed: {max_val - 1})")

    return position
