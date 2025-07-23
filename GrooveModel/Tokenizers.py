from dataclasses import dataclass

import torch

from GrooveModel.Utils.DNAOffset import encode_offset_ticks
from GrooveModel.Utils.DNAVelocity import encode_velocity
from GrooveModel.Utils.TimeSignatures import encode_time_signature
from GrooveModel.Utils.DNAGridFactor import encode_grid_factor
from GrooveModel.Utils.DNAValue import get_dna_instruments_list, encode_instrument

def trim_empty_measures(dna_units, grid_factor, numerator):
    empty_measures = 0
    measure_size = grid_factor * numerator  # units per measure
    for measure_index in range(0, len(dna_units), measure_size):
        measure = dna_units[measure_index:measure_index + measure_size]

        # Check if the entire measure is empty
        if all(unit['IsEmpty'] for unit in measure):
            empty_measures += 1
        else:
            break  # Stop trimming once we find a non-empty measure
    return dna_units[empty_measures * measure_size:]


@dataclass
class DNAToken:
    Instrument: int
    Velocity: int
    BeatUnit: int
    BeatUnitOffset: int
    GridFactor: int
    Bpm: int
    TimeSignature: int
    NumberOfBars: int
    TicksPerQuarter: int

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.Instrument,
            self.Velocity,
            self.BeatUnit,
            self.BeatUnitOffset,
            self.GridFactor,
            self.Bpm,
            self.TimeSignature,
            self.NumberOfBars,
            self.TicksPerQuarter,
        ], dtype=torch.int)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'DNAToken':
        return cls(*tensor.tolist())

def tokens_to_tensor(tokens: list[DNAToken]) -> torch.Tensor:
    return torch.stack([token.to_tensor() for token in tokens])

def tensor_to_tokens(tensor: torch.Tensor) -> list[DNAToken]:
    return [DNAToken.from_tensor(row) for row in tensor]

# TODO: add new encode functions
# TODO: Offset = +- ticks p gridUnit
# TODO:

class MultiDimDNATokenizer:
    @staticmethod
    def tokenize(song_json: dict, trim_leading_empty_measures: bool = True, absolute_grid_units: bool = False) -> list[DNAToken]:
        bpm = song_json.get('Bpm', 120)
        numerator = song_json.get('Numerator', 4)
        denominator = song_json.get('Denominator', 4)
        ticks_per_quarter = song_json.get('TicksPerQuarterNote', 480)
        grid_factor = song_json.get('GridFactor', 1)
        number_of_bars = song_json.get('NumberOfBars', 1)
        dna_units = song_json.get('DNAUnits', [])

        if trim_leading_empty_measures:
            dna_units = trim_empty_measures(dna_units, grid_factor, numerator)

        time_signature = encode_time_signature(numerator, denominator)
        grid_factor_encoded = encode_grid_factor(grid_factor)

        tokens = []

        for grid_unit, dna_unit in enumerate(dna_units):
            value = dna_unit.get('Value', 0)
            instruments = get_dna_instruments_list(value)
            velocities = dna_unit.get('VelocityPerValuePart', {})
            offsets = dna_unit.get('OffsetTicksPerValuePart', {})
            beat_unit = grid_unit if absolute_grid_units else grid_unit % (grid_factor * numerator)

            if not instruments:
                tokens.append(DNAToken(
                    Instrument=0,
                    Velocity=0,
                    BeatUnit=beat_unit,
                    BeatUnitOffset=0,
                    Bpm=bpm,
                    TimeSignature=time_signature,
                    GridFactor=grid_factor_encoded,
                    NumberOfBars=number_of_bars,
                    TicksPerQuarter=ticks_per_quarter,
                ))
                continue

            for instrument in instruments:
                velocity = encode_velocity(velocities.get(str(instrument), 0))
                offset = encode_offset_ticks(offsets.get(str(instrument), 0), ticks_per_quarter)

                tokens.append(DNAToken(
                    Instrument=encode_instrument(instrument),
                    Velocity=velocity,
                    BeatUnit=beat_unit,
                    BeatUnitOffset=offset,
                    Bpm=bpm,
                    TimeSignature=time_signature,
                    GridFactor=grid_factor_encoded,
                    NumberOfBars=number_of_bars,
                    TicksPerQuarter=ticks_per_quarter,
                ))

        return tokens
