from GrooveModel.Utils.SpecialTokens import SPECIAL_TOKEN_SIZE

# --- Config ---
MIN_BPM = 20
MAX_BPM = 300
NUM_BPM_BINS = 40

# --- Derived ---
_RANGE = MAX_BPM - MIN_BPM
_BIN_WIDTH = _RANGE / NUM_BPM_BINS
BPM_TOKEN_SIZE = NUM_BPM_BINS + SPECIAL_TOKEN_SIZE


def _to_bin_index(bpm: float) -> int:
    """
    Map a BPM to a bin index in [0, NUM_BPM_BINS-1].
    Bins are half-open: [low, high)
    """
    if not (MIN_BPM <= bpm < MAX_BPM):
        raise ValueError(f"BPM {bpm} out of range ({MIN_BPM}-{MAX_BPM})")

    # Normalize to [0, 1) then scale to bin count
    rel = (bpm - MIN_BPM) / _RANGE
    idx = int(rel * NUM_BPM_BINS)

    # Numerical safety (e.g., bpm extremely close to MAX_BPM)
    if idx >= NUM_BPM_BINS:
        idx = NUM_BPM_BINS - 1
    return idx


def encode_bpm(bpm: float, include_padding: bool = True) -> int:
    """
    Encode a BPM into a discrete bin token.
    """
    idx = _to_bin_index(bpm)
    return idx + SPECIAL_TOKEN_SIZE if include_padding else idx


def decode_bpm(token: int, include_padding: bool = True, as_int: bool = True) -> int | float:
    """
    Decode a bin token back to the midpoint BPM of that bin.
    If as_int=True, returns an int (rounded). Otherwise returns a float.
    """
    idx = token - SPECIAL_TOKEN_SIZE if include_padding else token
    if not (0 <= idx < NUM_BPM_BINS):
        raise ValueError(f"Token {token} maps to invalid bin index {idx} (0-{NUM_BPM_BINS - 1})")

    # Midpoint of bin k is: MIN + (k + 0.5) * BIN_WIDTH
    mid = MIN_BPM + (idx + 0.5) * _BIN_WIDTH
    return int(round(mid)) if as_int else mid


def bpm_bin_bounds(idx: int) -> tuple[int, int]:
    """
    Convenience: return (low, high) bounds for bin idx, half-open [low, high).
    """
    if not (0 <= idx < NUM_BPM_BINS):
        raise ValueError(f"Invalid bin index {idx} (0-{NUM_BPM_BINS - 1})")
    low = MIN_BPM + idx * _BIN_WIDTH
    high = low + _BIN_WIDTH
    return round(low), round(high)
