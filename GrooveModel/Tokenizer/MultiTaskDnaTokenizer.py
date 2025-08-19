from dataclasses import dataclass

import torch

from GrooveModel.Tokenizer.Tokenizer import DnaTokenizer, SongData
from GrooveModel.Utils.BeatUnit import encode_beat_unit
from GrooveModel.Utils.BeatsPerMinute import encode_bpm
from GrooveModel.Utils.DNAGridFactor import encode_grid_factor
from GrooveModel.Utils.DNAOffset import encode_offset_ticks
from GrooveModel.Utils.DNAValue import get_dna_instruments_list, encode_instrument, InstrumentValues
from GrooveModel.Utils.DNAVelocity import encode_velocity
from GrooveModel.Utils.TimeSignatures import encode_time_signature


@dataclass
class MultiDnaToken:
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
        ], dtype=torch.long)  # Long as it is required for loss calculation

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'MultiDnaToken':
        """Convert PyTorch tensor back into DNAToken."""
        return cls(*tensor.tolist())


# TODO: MAKE MORE PERFORMANT (DO ALL IN TENSOR SPACE ALREADY) -> OR Pretokenize

class MultiTaskDnaTokenizer(DnaTokenizer):

    @staticmethod
    def tokenize(song_json: dict, trim_leading_empty_measures: bool = True, absolute_grid_units: bool = False) -> list[
        MultiDnaToken]:
        """
        Tokenizes a song dictionary into a list of DNATokens.

        Each token represents a musical event or silent timestep, encoded as integers
        using special-token-aware functions for each attribute.
        """
        # Use shared extraction model
        song_data = SongData.from_json(song_json, trim_leading_empty_measures=trim_leading_empty_measures)

        # Encode song-level constants (still lives here)
        time_signature = encode_time_signature(song_data.numerator, song_data.denominator)
        grid_factor_encoded = encode_grid_factor(song_data.grid_factor)
        bpm_encoded = encode_bpm(song_data.bpm)

        tokens: list[MultiDnaToken] = []

        for grid_unit, dna_unit in enumerate(song_data.dna_units):
            value = dna_unit.get("Value", 0)
            instruments = get_dna_instruments_list(value)
            velocities = dna_unit.get("VelocityPerValuePart", {}) or {}
            offsets = dna_unit.get("OffsetTicksPerValuePart", {}) or {}

            beat_unit = song_data.beat_unit_for(grid_unit, absolute=absolute_grid_units)
            encoded_beat_unit = encode_beat_unit(beat_unit, absolute=absolute_grid_units)

            if not instruments:
                tokens.append(MultiDnaToken(
                    Instrument=encode_instrument(InstrumentValues.Rest),
                    Velocity=encode_velocity(0),
                    BeatUnit=encoded_beat_unit,
                    BeatUnitOffset=encode_offset_ticks(0, song_data.ticks_per_grid_unit),
                    Bpm=bpm_encoded,
                    TimeSignature=time_signature,
                    GridFactor=grid_factor_encoded,
                ))
                continue

            for instrument in instruments:
                encoded_velocity = encode_velocity(velocities.get(str(instrument), 0))
                encoded_offset = encode_offset_ticks(offsets.get(str(instrument), 0), song_data.ticks_per_grid_unit,
                                                     start_at_zero=True)

                tokens.append(MultiDnaToken(
                    Instrument=encode_instrument(instrument),
                    Velocity=encoded_velocity,
                    BeatUnit=encoded_beat_unit,
                    BeatUnitOffset=encoded_offset,
                    Bpm=bpm_encoded,
                    TimeSignature=time_signature,
                    GridFactor=grid_factor_encoded,
                ))

        return tokens

    @staticmethod
    def tokens_to_tensor(tokens: list[MultiDnaToken]) -> torch.Tensor:
        """Batch multiple DNATokens into a single tensor."""
        return torch.stack([token.to_tensor() for token in tokens])

    @staticmethod
    def tensor_to_tokens(tensor: torch.Tensor) -> list[MultiDnaToken]:
        """Convert a tensor back into a list of DNATokens."""
        return [MultiDnaToken.from_tensor(row) for row in tensor]
