import torch

from GrooveModel.Utils.SpecialTokens import SPECIAL_TOKEN_SIZE

# Canonical resolution (fixed MIDI velocity range 0–127)
VELOCITY_MIN = 0
VELOCITY_MAX = 127
VELOCITY_RESOLUTION = (VELOCITY_MAX - VELOCITY_MIN) + 1  # 128

# Effective resolution for token space (can be reduced, e.g. 64)
EFFECTIVE_VELOCITY_RESOLUTION = VELOCITY_RESOLUTION // 2
VELOCITY_TOKEN_SIZE = EFFECTIVE_VELOCITY_RESOLUTION + SPECIAL_TOKEN_SIZE


def _round_half_up(x: float) -> int:
    return int(x + 0.5)


def encode_velocity(
        velocity: float,
        resolution: int = VELOCITY_RESOLUTION,  # canonical calc resolution (always 128)
        effective_resolution: int = EFFECTIVE_VELOCITY_RESOLUTION,  # token resolution (e.g., 64)
        include_padding: bool = True
) -> int:
    """
    Quantize a velocity value in [0, 1] to a discrete index.
    - Uses 128-step canonical rounding first.
    - Projects to a coarser effective resolution for tokens.
    """
    if not (0.0 <= velocity <= 1.0):
        raise ValueError("velocity must be in [0, 1].")

    if effective_resolution < 2:
        raise ValueError("effective_resolution must be at least 2.")
    if resolution < 2:
        raise ValueError("resolution must be at least 2.")

    # Step 1: canonical rounding on the 128 grid (round half up)
    dna_step = 1.0 / (resolution - 1)
    steps_128 = _round_half_up(velocity / dna_step)
    steps_128 = max(0, min(steps_128, resolution - 1))

    # Step 2: project to effective grid
    if effective_resolution == resolution:
        steps_eff = steps_128
    else:
        steps_eff = _round_half_up(steps_128 * (effective_resolution - 1) / (resolution - 1))
        steps_eff = max(0, min(steps_eff, effective_resolution - 1))

    return steps_eff + SPECIAL_TOKEN_SIZE if include_padding else steps_eff


def decode_velocity(
        encoded_velocity: int,
        resolution: int = VELOCITY_RESOLUTION,  # canonical calc resolution
        effective_resolution: int = EFFECTIVE_VELOCITY_RESOLUTION,  # must match encoding
        include_padding: bool = True
) -> float:
    """
    Decode a velocity index back into a normalized [0, 1] float.
    """

    if effective_resolution < 2:
        raise ValueError("effective_resolution must be at least 2.")
    if resolution < 2:
        raise ValueError("resolution must be at least 2.")

    index = encoded_velocity - SPECIAL_TOKEN_SIZE if include_padding else encoded_velocity

    if not (0 <= index <= effective_resolution - 1):
        raise ValueError(f"encoded index out of range for effective_resolution={effective_resolution}.")

    return index / float(effective_resolution - 1)


def normalize_velocity_tensor(
        vel_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    vel_idx = (vel_ids - SPECIAL_TOKEN_SIZE).clamp(min=0, max=EFFECTIVE_VELOCITY_RESOLUTION - 1)
    vel_tensor = vel_idx.to(dtype) / float(max(1, EFFECTIVE_VELOCITY_RESOLUTION - 1))
    return vel_tensor
