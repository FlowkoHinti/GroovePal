from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

import torch


# Import all domain-specific encoding utilities


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
    dna_units: list[Dict]  # raw grid step dicts

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
    def tokenize(song_json, **kwargs) -> list[Any] | torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def tokens_to_tensor(tokens: list[Any]) -> torch.Tensor:
        """Batch multiple DNATokens into a single tensor."""
        pass

    @staticmethod
    @abstractmethod
    def tensor_to_tokens(tensor: torch.Tensor) -> list[Any]:
        """Convert a tensor back into a list of DNATokens."""
        pass
