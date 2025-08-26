import json
from typing import ClassVar

import torch

from GrooveModel.Tokenizer.Tokenizer import DnaTokenizer, SongData
from GrooveModel.Utils.BeatsPerMinute import encode_bpm, bpm_bin_bounds
from GrooveModel.Utils.DNAGridFactor import GridFactors
from GrooveModel.Utils.DNAOffset import encode_offset_ticks
from GrooveModel.Utils.DNAValue import get_dna_instruments_list, encode_instrument
from GrooveModel.Utils.DNAVelocity import encode_velocity
from GrooveModel.Vocab import SequentialDnaVocab


class SequentialDnaTokenizer(DnaTokenizer):
    vocab: ClassVar[SequentialDnaVocab] = SequentialDnaVocab()

    @staticmethod
    def tokenize(song_json, trim_leading_empty_measures: bool = True,
                 absolute_grid_units: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        song_data = SongData.from_json(song_json, trim_leading_empty_measures=trim_leading_empty_measures)
        bpm_encoded = encode_bpm(song_data.bpm, include_padding=False)

        tokens: list[str] = []
        beat_positions: list[int] = []

        meta_sequence = [f'BOS',
                         f'Bpm_bin_{bpm_encoded}',
                         f'Time_{song_data.numerator}_{song_data.denominator}',
                         f'{GridFactors(song_data.grid_factor).name}']
        tokens.extend(meta_sequence)


        # Add positional mappings for meta
        beat_positions.extend([0] * len(meta_sequence))

        for grid_unit, dna_unit in enumerate(song_data.dna_units):
            value = dna_unit.get("Value", 0)
            instruments = get_dna_instruments_list(value)
            velocities = dna_unit.get("VelocityPerValuePart", {})
            offsets = dna_unit.get("OffsetTicksPerValuePart", {})

            # Calculate current beat unit (grid position)
            beat_unit = song_data.beat_unit_for(grid_unit, absolute=absolute_grid_units)

            # -> sentence start token?

            # If the step is empty (no instruments), add a "rest" token
            if not instruments:
                rest_sequence = [f'Rest', f'Vel_{0}', f'Off_{0}' f'SEP']
                tokens.extend(rest_sequence)
                beat_positions.extend([beat_unit] * len(rest_sequence))
                continue

            # Otherwise, create a token for each active instrument
            for instrument in instruments:
                encoded_instrument = encode_instrument(instrument).name
                encoded_velocity = encode_velocity(velocities.get(str(instrument), 0), include_padding=False)
                encoded_offset = encode_offset_ticks(offsets.get(str(instrument), 0), song_data.ticks_per_grid_unit,
                                                     include_padding=False)
                instrument_sequence = [f'{encoded_instrument}', f'Vel_{encoded_velocity}', f'Off_{encoded_offset}']
                tokens.extend(instrument_sequence)
                beat_positions.extend([beat_unit] * len(instrument_sequence))

            tokens.append(f'SEP')
            beat_positions.append(beat_unit)

        tokens.append(f'EOS')
        beat_positions.append(beat_positions[-1])

        token_ids = [SequentialDnaTokenizer.vocab[t] for t in tokens]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(beat_positions, dtype=torch.long)




test_json = f'../../Data/unit_test/unit_test_chunk_1.jsonl'

with open(test_json, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            SequentialDnaTokenizer.tokenize(json.loads(line), trim_leading_empty_measures=True)





# TODO: make it cleaner and faster
# TODO: truncate and padding
# TODO: Bin values -> also for other version
# TODO: Positional encoding -> handle according to SEP
# TODO: Embedding
# TODO: Model and Learner
# TODO: Define experiments
# TODO: Pretrained xlstm?
