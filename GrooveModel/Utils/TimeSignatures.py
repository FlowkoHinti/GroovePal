TIME_SIGNATURE_LOOKUP = {
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
    (3, 2): 11
    # Add more as needed
}

# Optional reverse lookup
ID_TO_TIME_SIGNATURE = {v: k for k, v in TIME_SIGNATURE_LOOKUP.items()}

TIME_SIGNATURES_SIZE = len(TIME_SIGNATURE_LOOKUP)

UNKNOWN_TIME_SIGNATURE_ID = TIME_SIGNATURES_SIZE



def encode_time_signature(numerator: int, denominator: int) -> int:
    """
    Encodes the time signature into an ID, starting denominators at 2.
    """
    if denominator < 2:
        raise ValueError("Denominator must be >= 2.")
    return TIME_SIGNATURE_LOOKUP.get((numerator, denominator), UNKNOWN_TIME_SIGNATURE_ID)

def decode_time_signature(time_id: int) -> (int, int):
    """
    Decodes the time signature ID back to (numerator, denominator).
    """
    if time_id not in ID_TO_TIME_SIGNATURE:
        raise ValueError(f"Unknown time signature ID: {time_id}")
    return ID_TO_TIME_SIGNATURE[time_id]