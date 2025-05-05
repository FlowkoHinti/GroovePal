import numpy as np
from math import floor, ceil

OFFSET_TICKS_SIZE = 128
HALF_OFFSET_TICKS = OFFSET_TICKS_SIZE / 2
QUARTER_OFFSET_TICKS = OFFSET_TICKS_SIZE / 4
EIGHTH_OFFSET_TICKS = OFFSET_TICKS_SIZE / 8

def normalize_offset_ticks(offset: int, ticks_per_qn: int, resolution: int = OFFSET_TICKS_SIZE) -> int:
    """
    Convert the dna offset value to standard range.
    :param offset: The offset value to convert.
    :param ticks_per_qn: The number of ticks per quarter note.
    :param resolution: The resolution for the offset value (how many steps to divide the range into).
    :return: The converted integer value.
    """
    negative = offset < 0
    offset = abs(offset)
    dna_step = ticks_per_qn / resolution
    steps = floor(offset / dna_step)
    remainder = offset % dna_step
    if remainder >= dna_step / 2:
        steps += 1
    return -steps if negative else steps


# TODO: add optional random sampling of the offset when ticks per qn is a lot greater than the normed resolution
def denormalize_offset_ticks(norm_offset: int, ticks_per_qn: int, norm_resolution: int = OFFSET_TICKS_SIZE, randomizer: bool = False) -> int:
    denormalized_offset = normalize_offset_ticks(norm_offset, ticks_per_qn, resolution=norm_resolution)
    dna_step = ticks_per_qn / norm_resolution
    random_deviation = 0
    if randomizer:
        #Sample a random deviation from the offset
        random_deviation = np.random.randint(#TODO)
    return denormalized_offset + random_deviation