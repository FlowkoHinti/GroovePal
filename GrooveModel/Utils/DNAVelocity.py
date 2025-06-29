from math import floor

# According to MIDI standard, the velocity range is from 0 to 127.
VELOCITY_MIN = 0
VELOCITY_MAX = 127
VELOCITY_SIZE = (VELOCITY_MAX - VELOCITY_MIN) + 1

# Reduced velocity ranges
HALF_VELOCITY_SIZE = VELOCITY_SIZE // 2
QUARTER_VELOCITY_SIZE = VELOCITY_SIZE // 4
EIGHTH_VELOCITY_SIZE = VELOCITY_SIZE // 8

def normalize_velocity(velocity: float, resolution: int = VELOCITY_SIZE) -> int:
    """
    Convert the dna velocity value to standard MIDI velocity range.
    :param velocity: The velocity value to convert (expected in range [0, 1]).
    :param resolution: Number of velocity steps (default 128 for MIDI).
    :return: The quantized velocity as an integer (0 to resolution - 1).
    """

    dna_step = 1 / (resolution - 1)
    steps = floor(velocity / dna_step)
    remainder = velocity % dna_step
    if remainder >= dna_step / 2:
        steps += 1

    # Clamp to maximum valid step
    return min(steps, resolution - 1)
