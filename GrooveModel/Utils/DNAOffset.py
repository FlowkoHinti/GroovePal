import torch

from GrooveModel.Utils.SpecialTokens import SPECIAL_TOKEN_SIZE

OFFSET_TICKS_RESOLUTION = 120  # usable steps excluding special tokens
OFFSET_TOKEN_SIZE = OFFSET_TICKS_RESOLUTION + SPECIAL_TOKEN_SIZE  # total size including special tokens

DEFAULT_PERCENT_STEP = 0.05  # 5% per step
OFFSET_STEPS = int(1 // DEFAULT_PERCENT_STEP)


def encode_offset_ticks(offset: int, ticks_per_grid_unit: int = OFFSET_TICKS_RESOLUTION, start_at_zero=False,
                        resolution: int = OFFSET_TICKS_RESOLUTION, include_padding: bool = True) -> int:
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
                        resolution: int = OFFSET_TICKS_RESOLUTION, include_padding: bool = True) -> int:
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


def offset_to_percent_step(
    offset: int,
    ticks_per_grid_unit: int = OFFSET_TICKS_RESOLUTION,
    start_at_zero: bool = False,
    percent_step: float = DEFAULT_PERCENT_STEP,
) -> int:
    """
    Map a tick `offset` to a percentage *step id*.

    - If start_at_zero=False: steps span [-100%, +100%] with size `percent_step*100%`.
      Returned step ids are in [-1/percent_step, +1/percent_step].
    - If start_at_zero=True: steps span [0%, 100%] with size `percent_step*100%`.
      Returned step ids are in [0, +1/percent_step].

    Example: percent_step=0.05 (5%) -> ids in [-20..+20] (or [0..20] if start_at_zero=True).
    """
    if not (0.0 < percent_step <= 1.0):
        raise ValueError("percent_step must be in (0, 1]. For 5% use 0.05.")

    # normalized v: [-1,1] or [0,1]
    v = normalize_offset(offset, ticks_per_grid_unit=ticks_per_grid_unit, start_at_zero=start_at_zero)

    # convert to step id on the normalized scale (so 1.0 == 100%)
    step_id = int(round(v / percent_step))

    # clamp to legal range
    max_step = int(round(1.0 / percent_step))
    if start_at_zero:
        step_id = max(0, min(max_step, step_id))
    else:
        step_id = max(-max_step, min(max_step, step_id))

    return step_id


def percent_step_to_offset(
    step_id: int,
    ticks_per_grid_unit: int = OFFSET_TICKS_RESOLUTION,
    start_at_zero: bool = False,
    percent_step: float = DEFAULT_PERCENT_STEP,
) -> int:
    """
    Map a percentage *step id* back to a tick `offset`.

    - If start_at_zero=False: legal step_id in [-1/percent_step, +1/percent_step].
    - If start_at_zero=True: legal step_id in [0, +1/percent_step].

    Each step is `percent_step*100%`. For example, percent_step=0.05 -> each step = 5%.
    """
    if not (0.0 < percent_step <= 1.0):
        raise ValueError("percent_step must be in (0, 1]. For 5% use 0.05.")

    max_step = int(round(1.0 / percent_step))
    if start_at_zero:
        step_id = max(0, min(max_step, step_id))
    else:
        step_id = max(-max_step, min(max_step, step_id))

    # back to normalized value v in [-1,1] or [0,1]
    v = step_id * percent_step

    # and finally to ticks
    return denormalize_offset(v, ticks_per_grid_unit=ticks_per_grid_unit, start_at_zero=start_at_zero)



def normalize_offset(offset: int, ticks_per_grid_unit: int = OFFSET_TICKS_RESOLUTION,
                     start_at_zero: bool = False) -> float:
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


def denormalize_offset(norm_offset: float, ticks_per_grid_unit: int = OFFSET_TICKS_RESOLUTION,
                       start_at_zero: bool = False) -> int:
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
