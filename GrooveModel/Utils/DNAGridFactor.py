from enum import IntEnum, auto

from GrooveModel.Utils.SpecialTokens import SPECIAL_TOKEN_SIZE


class GridFactors(IntEnum):
    Quarter = 1
    Eighth = 2
    Sixteenth = 4
    EighthTriplet = 3
    SixteenthTriplet = 6


class RemappedGridFactors(IntEnum):
    Quarter = SPECIAL_TOKEN_SIZE
    Eighth = auto()
    Sixteenth = auto()
    EighthTriplet = auto()
    SixteenthTriplet = auto()


GRID_FACTOR_TOKEN_SIZE = len(GridFactors) + SPECIAL_TOKEN_SIZE


def encode_grid_factor(grid_factor) -> RemappedGridFactors:
    dna_grid_factor = GridFactors(grid_factor).name
    return RemappedGridFactors[dna_grid_factor]

def decode_grid_factor(grid_factor) -> GridFactors:
    dna_grid_factor = RemappedGridFactors(grid_factor).name
    return GridFactors[dna_grid_factor]
