from enum import IntEnum, auto

from GrooveModel.Utils.SpecialTokens import SPECIAL_TOKEN_SIZE


class TimeSignatures(IntEnum):
    Time_4_4 = 0
    Time_3_4 = 1
    Time_6_8 = 2
    Time_2_4 = 3
    Time_2_2 = 4
    Time_5_4 = 5
    Time_7_8 = 6
    Time_9_8 = 7
    Time_12_8 = 8
    Time_3_8 = 9
    Time_6_4 = 10
    Time_3_2 = 11
    Time_Unknown = 12  # Add this explicitly


class RemappedTimeSignatures(IntEnum):
    Time_4_4 = SPECIAL_TOKEN_SIZE
    Time_3_4 = auto()
    Time_6_8 = auto()
    Time_2_4 = auto()
    Time_2_2 = auto()
    Time_5_4 = auto()
    Time_7_8 = auto()
    Time_9_8 = auto()
    Time_12_8 = auto()
    Time_3_8 = auto()
    Time_6_4 = auto()
    Time_3_2 = auto()
    Time_Unknown = auto()  # Ensure a clean symbolic ID for unknown


TIME_SIGNATURE_TOKEN_SIZE = len(TimeSignatures) + SPECIAL_TOKEN_SIZE
UNKNOWN_TIME_SIGNATURE_ID = RemappedTimeSignatures.Time_Unknown

_TIME_SIGNATURE_NAME_LOOKUP = {
    (4, 4): "Time_4_4",
    (3, 4): "Time_3_4",
    (6, 8): "Time_6_8",
    (2, 4): "Time_2_4",
    (2, 2): "Time_2_2",
    (5, 4): "Time_5_4",
    (7, 8): "Time_7_8",
    (9, 8): "Time_9_8",
    (12, 8): "Time_12_8",
    (3, 8): "Time_3_8",
    (6, 4): "Time_6_4",
    (3, 2): "Time_3_2",
    # Add more if needed
}

ID_TO_TIME_SIGNATURE = {
    RemappedTimeSignatures[name]: (numerator, denominator)
    for (numerator, denominator), name in _TIME_SIGNATURE_NAME_LOOKUP.items()
}
ID_TO_TIME_SIGNATURE[RemappedTimeSignatures.Time_Unknown] = ("?", "?")  # symbolic fallback


def encode_time_signature(numerator: int, denominator: int) -> RemappedTimeSignatures:
    """
    Encodes a time signature (numerator, denominator) to a remapped enum ID.
    Returns RemappedTimeSignatures.Unknown if the time signature is not recognized.
    """
    if denominator < 2:
        raise ValueError("Denominator must be >= 2.")

    name = _TIME_SIGNATURE_NAME_LOOKUP.get((numerator, denominator))
    if name is None:
        return RemappedTimeSignatures.Time_Unknown

    return RemappedTimeSignatures[name]


def decode_time_signature(time_id: int) -> tuple[int, int] | tuple[str, str]:
    """
    Decodes a remapped time signature token ID back to (numerator, denominator).
    Returns ("?", "?") if the ID is unknown.
    """
    return ID_TO_TIME_SIGNATURE.get(time_id, ("?", "?"))
