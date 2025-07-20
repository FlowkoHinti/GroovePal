from GrooveModel.Utils.SpecialTokens import SPECIAL_TOKEN_SIZE

# Raw mapping starts from 0 internally
_RAW_TIME_SIGNATURE_LOOKUP = {
    (4, 4): 0,
    (3, 4): 1,
    (6, 8): 2,
    (2, 4): 3,
    (2, 2): 4,
    (5, 4): 5,
    (7, 8): 6,
    (9, 8): 7,
    (12, 8): 8,
    (3, 8): 9,
    (6, 4): 10,
    (3, 2): 11,
    # Add more as needed
}

# Time signatures start after the special token range
TIME_SIGNATURE_LOOKUP = {
    k: v + SPECIAL_TOKEN_SIZE for k, v in _RAW_TIME_SIGNATURE_LOOKUP.items()
}

ID_TO_TIME_SIGNATURE = {
    v: k for k, v in TIME_SIGNATURE_LOOKUP.items()
}

# Size of real values (excluding specials)
TIME_SIGNATURE_RESOLUTION = len(_RAW_TIME_SIGNATURE_LOOKUP)

# Full token vocabulary size
TIME_SIGNATURE_TOKEN_SIZE = TIME_SIGNATURE_RESOLUTION + SPECIAL_TOKEN_SIZE

# ID for unknown time signatures
UNKNOWN_TIME_SIGNATURE_ID = TIME_SIGNATURE_TOKEN_SIZE


def encode_time_signature(numerator: int, denominator: int) -> int:
    """
    Encodes a time signature (numerator, denominator) to a token ID.
    Time signature tokens start at SPECIAL_TOKEN_SIZE.
    """
    if denominator < 2:
        raise ValueError("Denominator must be >= 2.")
    return TIME_SIGNATURE_LOOKUP.get((numerator, denominator), UNKNOWN_TIME_SIGNATURE_ID)


def decode_time_signature(time_id: int) -> tuple[int, int]:
    """
    Decodes a token ID back to (numerator, denominator).
    """
    if time_id not in ID_TO_TIME_SIGNATURE:
        raise ValueError(f"Unknown time signature ID: {time_id}")
    return ID_TO_TIME_SIGNATURE[time_id]