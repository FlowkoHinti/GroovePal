from math import floor

# According to MIDI standard, the velocity range is from 0 to 127.
VELOCITY_MIN = 0
VELOCITY_MAX = 127
VELOCITY_SIZE = VELOCITY_MAX - VELOCITY_MIN + 1

# Reduced velocity ranges
HALF_VELOCITY_SIZE = VELOCITY_SIZE/2
QUARTER_VELOCITY_SIZE = VELOCITY_SIZE/4
EIGHTH_VELOCITY_SIZE = VELOCITY_SIZE/8

def normalize_velocity(velocity: float, resolution: int = VELOCITY_SIZE) -> int:
    """
    Convert the dna velocity value to standard range.
    :param velocity: The velocity value to convert.
    :param resolution: The resolution for the velocity value (how many steps to divide the range into).
    :return: The converted integer value.
    """

    dna_step = 1 / resolution
    steps = floor(velocity / dna_step)
    remainder = velocity % dna_step
    if remainder >= dna_step / 2:
        steps += 1
    return steps
