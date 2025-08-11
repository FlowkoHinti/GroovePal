from math import floor
from GrooveModel.Utils.SpecialTokens import SPECIAL_TOKEN_SIZE

# TODO: MAYBE REDUCE RANGE -> LESS VALUES TO PREDICT

VELOCITY_MIN = 0
VELOCITY_MAX = 127
VELOCITY_RESOLUTION = (VELOCITY_MAX - VELOCITY_MIN) + 1  # 128
VELOCITY_TOKEN_SIZE = VELOCITY_RESOLUTION + SPECIAL_TOKEN_SIZE


def encode_velocity(velocity: float, resolution: int = VELOCITY_RESOLUTION) -> int:
    """
    Quantize a velocity value in [0, 1] into a discrete velocity index with special token offset.
    :param velocity: Normalized float in [0, 1].
    :param resolution: Number of steps (default 128).
    :return: Encoded velocity index with SPECIAL_TOKEN_SIZE offset.
    """
    dna_step = 1 / (resolution - 1)
    steps = floor(velocity / dna_step)
    remainder = velocity % dna_step
    if remainder >= dna_step / 2:
        steps += 1

    steps = min(steps, resolution - 1)
    return steps + SPECIAL_TOKEN_SIZE


def decode_velocity(encoded_velocity: int, resolution: int = VELOCITY_RESOLUTION) -> float:
    """
    Decode a velocity index back into a normalized [0, 1] float.
    :param encoded_velocity: Encoded velocity index including SPECIAL_TOKEN_SIZE offset.
    :param resolution: Number of steps used (default 128).
    :return: Normalized float velocity in [0, 1].
    """
    index = encoded_velocity - SPECIAL_TOKEN_SIZE
    return index / (resolution - 1)