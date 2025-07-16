from enum import IntEnum

class GridFactors(IntEnum):
    Quarter = 1
    Eighth = 2
    Sixteenth = 4
    EighthTriplet = 3
    SixteenthTriplet = 6

class RemappedGridFactors(IntEnum):
    Quarter = 0
    Eighth = 1
    Sixteenth = 2
    EighthTriplet = 3
    SixteenthTriplet = 4

GRID_FACTORS_SIZE = sum([value.value for value in GridFactors]) + 1

def encode_grid_factor(grid_factor) -> RemappedGridFactors:
    dna_grid_factor = GridFactors(grid_factor).name
    return RemappedGridFactors[dna_grid_factor]

def decode_grid_factor(grid_factor) -> GridFactors:
    dna_grid_factor = RemappedGridFactors(grid_factor).name
    return GridFactors[dna_grid_factor]