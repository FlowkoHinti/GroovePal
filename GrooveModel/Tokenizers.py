from dataclasses import dataclass
import torch

# Import all domain-specific encoding utilities
from GrooveModel.Utils.BeatUnit import encode_beat_unit
from GrooveModel.Utils.BeatsPerMinute import encode_bpm
from GrooveModel.Utils.DNAOffset import encode_offset_ticks
from GrooveModel.Utils.DNAVelocity import encode_velocity
from GrooveModel.Utils.TimeSignatures import encode_time_signature
from GrooveModel.Utils.DNAGridFactor import encode_grid_factor
from GrooveModel.Utils.DNAValue import get_dna_instruments_list, encode_instrument, InstrumentValues


def trim_empty_measures(dna_units, grid_factor, numerator):
    """
    Removes empty measures at the beginning of the sequence.

    A measure is considered empty if all its DNA units have the 'IsEmpty' flag set.
    This helps avoid training on leading silence.
    """
    empty_measures = 0
    measure_size = grid_factor * numerator  # total grid units per measure

    for measure_index in range(0, len(dna_units), measure_size):
        measure = dna_units[measure_index:measure_index + measure_size]

        if all(unit['IsEmpty'] for unit in measure):  # skip silent measures
            empty_measures += 1
        else:
            break  # stop trimming once actual data starts

    return dna_units[empty_measures * measure_size:]


@dataclass
class DNAToken:
    """
    Represents a single token of musical information for a timestep/grid unit.
    """
    Instrument: int
    Velocity: int
    BeatUnit: int
    BeatUnitOffset: int
    GridFactor: int
    Bpm: int
    TimeSignature: int

    def to_tensor(self) -> torch.Tensor:
        """Convert DNAToken to PyTorch tensor."""
        return torch.tensor([
            self.Instrument,
            self.Velocity,
            self.BeatUnit,
            self.BeatUnitOffset,
            self.GridFactor,
            self.Bpm,
            self.TimeSignature,
        ], dtype=torch.long) # Long as it is required for loss calculation

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'DNAToken':
        """Convert PyTorch tensor back into DNAToken."""
        return cls(*tensor.tolist())


def tokens_to_tensor(tokens: list[DNAToken]) -> torch.Tensor:
    """Batch multiple DNATokens into a single tensor."""
    return torch.stack([token.to_tensor() for token in tokens])


def tensor_to_tokens(tensor: torch.Tensor) -> list[DNAToken]:
    """Convert a tensor back into a list of DNATokens."""
    return [DNAToken.from_tensor(row) for row in tensor]


class MultiTaskDNATokenizer:
    @staticmethod
    def tokenize(song_json: dict, trim_leading_empty_measures: bool = True, absolute_grid_units: bool = False) -> list[DNAToken]:
        """
        Tokenizes a song dictionary into a list of DNATokens.

        Each token represents a musical event or silent timestep, encoded as integers
        using special-token-aware functions for each attribute.
        """
        # Extract song-level metadata
        bpm = song_json.get('Bpm', 120)
        numerator = song_json.get('Numerator', 4)
        denominator = song_json.get('Denominator', 4)
        grid_factor = song_json.get('GridFactor', 1)

        # Use explicit TicksPerGridUnit if given, or compute from TicksPerQuarter
        ticks_per_grid_unit = song_json.get(
            'TicksPerGridUnit',
            song_json.get('TicksPerQuarterNote', 480) // grid_factor
        )

        dna_units = song_json.get('DNAUnits', [])

        # Remove leading empty bars if configured
        if trim_leading_empty_measures:
            dna_units = trim_empty_measures(dna_units, grid_factor, numerator)

        # Encode fixed song-level metadata into token-compatible values
        time_signature = encode_time_signature(numerator, denominator)
        grid_factor_encoded = encode_grid_factor(grid_factor)
        bpm_encoded = encode_bpm(bpm)

        tokens = []

        # Loop through each grid step (unit) in the sequence
        for grid_unit, dna_unit in enumerate(dna_units):
            value = dna_unit.get('Value', 0)
            instruments = get_dna_instruments_list(value)
            velocities = dna_unit.get('VelocityPerValuePart', {})
            offsets = dna_unit.get('OffsetTicksPerValuePart', {})

            # Calculate current beat unit (grid position)
            beat_unit = grid_unit if absolute_grid_units else grid_unit % (grid_factor * numerator)
            encoded_beat_unit = encode_beat_unit(beat_unit, absolute=absolute_grid_units)

            # If the step is empty (no instruments), add a "rest" token
            if not instruments:
                tokens.append(DNAToken(
                    Instrument=encode_instrument(InstrumentValues.Rest),
                    Velocity=encode_velocity(0),
                    BeatUnit=encoded_beat_unit,
                    BeatUnitOffset=encode_offset_ticks(0, ticks_per_grid_unit),
                    Bpm=bpm_encoded,
                    TimeSignature=time_signature,
                    GridFactor=grid_factor_encoded,
                ))
                continue

            # Otherwise, create a token for each active instrument
            for instrument in instruments:
                encoded_velocity = encode_velocity(velocities.get(str(instrument), 0))
                encoded_offset = encode_offset_ticks(offsets.get(str(instrument), 0), ticks_per_grid_unit)

                tokens.append(DNAToken(
                    Instrument=encode_instrument(instrument),
                    Velocity=encoded_velocity,
                    BeatUnit=encoded_beat_unit,
                    BeatUnitOffset=encoded_offset,
                    Bpm=bpm_encoded,
                    TimeSignature=time_signature,
                    GridFactor=grid_factor_encoded,
                ))

        return tokens