from enum import IntEnum, auto

from GrooveModel.Utils.SpecialTokens import SPECIAL_TOKEN_SIZE


class GridFactors(IntEnum):
    Quarter_Grid = 1
    Eighth_Grid = 2
    Sixteenth_Grid = 4
    EighthTriplet_Grid = 3
    SixteenthTriplet_Grid = 6


class RemappedGridFactors(IntEnum):
    Quarter_Grid = SPECIAL_TOKEN_SIZE
    Eighth_Grid = auto()
    Sixteenth_Grid = auto()
    EighthTriplet_Grid = auto()
    SixteenthTriplet_Grid = auto()


GRID_FACTOR_TOKEN_SIZE = len(GridFactors) + SPECIAL_TOKEN_SIZE


def encode_grid_factor(grid_factor) -> RemappedGridFactors:
    dna_grid_factor = GridFactors(grid_factor).name
    return RemappedGridFactors[dna_grid_factor]

def decode_grid_factor(grid_factor) -> GridFactors:
    dna_grid_factor = RemappedGridFactors(grid_factor).name
    return GridFactors[dna_grid_factor]
