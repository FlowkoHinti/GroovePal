from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

import torch

# Import all domain-specific encoding utilities
from GrooveModel.Utils.BeatUnit import encode_beat_unit
from GrooveModel.Utils.BeatsPerMinute import encode_bpm
from GrooveModel.Utils.DNAGridFactor import encode_grid_factor, GridFactors
from GrooveModel.Utils.DNAOffset import encode_offset_ticks
from GrooveModel.Utils.DNAValue import get_dna_instruments_list, encode_instrument, InstrumentValues
from GrooveModel.Utils.DNAVelocity import encode_velocity
from GrooveModel.Utils.TimeSignatures import encode_time_signature
from GrooveModel.Vocab import SequentialDnaVocab


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


@dataclass(frozen=True)
class SongData:
    bpm: int
    numerator: int
    denominator: int
    grid_factor: int
    ticks_per_grid_unit: int
    dna_units: List[Dict]  # raw grid step dicts

    @classmethod
    def from_json(
            cls,
            song_json: Dict,
            *,
            trim_leading_empty_measures: bool = True,
    ) -> SongData:
        """Extract raw song data without encoding anything."""
        bpm = int(song_json.get("Bpm", 120))
        numerator = int(song_json.get("Numerator", 4))
        denominator = int(song_json.get("Denominator", 4))
        grid_factor = int(song_json.get("GridFactor", 1))

        ticks_per_grid_unit = int(song_json.get(
            "TicksPerGridUnit",
            song_json.get("TicksPerQuarterNote", 480) // max(grid_factor, 1)
        ))

        dna_units = list(song_json.get("DNAUnits", []))

        if trim_leading_empty_measures:
            dna_units = trim_empty_measures(dna_units, grid_factor, numerator)

        return cls(
            bpm=bpm,
            numerator=numerator,
            denominator=denominator,
            grid_factor=grid_factor,
            ticks_per_grid_unit=ticks_per_grid_unit,
            dna_units=dna_units
        )

    def beat_unit_for(self, grid_unit: int, *, absolute: bool) -> int:
        return grid_unit if absolute else grid_unit % (self.grid_factor * self.numerator)


class DnaTokenizer(ABC):
    @staticmethod
    @abstractmethod
    def tokenize(song_json):
        pass


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


def tokens_to_tensor(tokens: list[MultiDnaToken]) -> torch.Tensor:
    """Batch multiple DNATokens into a single tensor."""
    return torch.stack([token.to_tensor() for token in tokens])


def tensor_to_tokens(tensor: torch.Tensor) -> list[MultiDnaToken]:
    """Convert a tensor back into a list of DNATokens."""
    return [MultiDnaToken.from_tensor(row) for row in tensor]


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
                encoded_offset = encode_offset_ticks(offsets.get(str(instrument), 0), song_data.ticks_per_grid_unit)

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


class SequentialDnaTokenizer(DnaTokenizer):
    @staticmethod
    def tokenize(song_json, trim_leading_empty_measures: bool = True, absolute_grid_units: bool = False) -> torch.Tensor:

        vocab = SequentialDnaVocab()
        song_data = SongData.from_json(song_json, trim_leading_empty_measures=trim_leading_empty_measures)

        tokens: list[str] = []

        tokens.append(f'BOS')
        tokens.append(f'BPM_{song_data.bpm}')
        tokens.append(f'Time_{song_data.numerator}_{song_data.denominator}')
        tokens.append(f'{GridFactors(song_data.grid_factor).name}')
        # -> sentence start token?

        for grid_unit, dna_unit in enumerate(song_data.dna_units):
            value = dna_unit.get("Value", 0)
            instruments = get_dna_instruments_list(value)
            velocities = dna_unit.get("VelocityPerValuePart", {})
            offsets = dna_unit.get("OffsetTicksPerValuePart", {})


            # Calculate current beat unit (grid position)
            beat_unit = song_data.beat_unit_for(grid_unit, absolute=absolute_grid_units)

            # If the step is empty (no instruments), add a "rest" token
            if not instruments:
                tokens.append(f'Rest')
                tokens.append(f'VEL_{0}')
                tokens.append(f'OFF_{0}')
                tokens.append(f'SEP')
                continue

            # Otherwise, create a token for each active instrument
            for instrument in instruments:
                encoded_instrument = encode_instrument(instrument).name
                encoded_velocity = encode_velocity(velocities.get(str(instrument), 0), include_padding=False)
                encoded_offset = encode_offset_ticks(offsets.get(str(instrument), 0), song_data.ticks_per_grid_unit, include_padding=False)

                tokens.append(f'{encoded_instrument}')
                tokens.append(f'VEL_{encoded_velocity}')
                tokens.append(f'OFF_{encoded_offset}')
            tokens.append(f'SEP')
        tokens.append(f'EOS')

        token_ids = [vocab[t] for t in tokens]
        return torch.tensor(token_ids, dtype=torch.long)


test_json = '../Data/unit_test/dnas.json'
with open(test_json, 'r') as f:
    test_json = json.load(f)

test_json = test_json[0]
SequentialDnaTokenizer.tokenize(test_json, trim_leading_empty_measures=True)

#TODO: make it cleaner and faster
#TODO: truncate and padding
#TODO: Bin values -> also for other version
#TODO: Positional encoding -> handle according to SEP
#TODO: Embedding
#TODO: Model and Learner
#TODO: Define experiments
#TODO: Pretrained xlstm?
