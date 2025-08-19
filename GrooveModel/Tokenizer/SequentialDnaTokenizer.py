import torch

from GrooveModel.Tokenizer.Tokenizer import DnaTokenizer, SongData
from GrooveModel.Utils.DNAGridFactor import GridFactors
from GrooveModel.Utils.DNAOffset import encode_offset_ticks
from GrooveModel.Utils.DNAValue import get_dna_instruments_list, encode_instrument
from GrooveModel.Utils.DNAVelocity import encode_velocity
from GrooveModel.Vocab import SequentialDnaVocab


class SequentialDnaTokenizer(DnaTokenizer):
    @staticmethod
    def tokenize(song_json, trim_leading_empty_measures: bool = True,
                 absolute_grid_units: bool = False) -> torch.Tensor:

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
                encoded_offset = encode_offset_ticks(offsets.get(str(instrument), 0), song_data.ticks_per_grid_unit,
                                                     include_padding=False)

                tokens.append(f'{encoded_instrument}')
                tokens.append(f'VEL_{encoded_velocity}')
                tokens.append(f'OFF_{encoded_offset}')
            tokens.append(f'SEP')
        tokens.append(f'EOS')

        token_ids = [vocab[t] for t in tokens]
        return torch.tensor(token_ids, dtype=torch.long)

# test_json = '../../Data/unit_test/dnas.json'
# with open(test_json, 'r') as f:
#     test_json = json.load(f)
#
# test_json = test_json[0]
# SequentialDnaTokenizer.tokenize(test_json, trim_leading_empty_measures=True)

# TODO: make it cleaner and faster
# TODO: truncate and padding
# TODO: Bin values -> also for other version
# TODO: Positional encoding -> handle according to SEP
# TODO: Embedding
# TODO: Model and Learner
# TODO: Define experiments
# TODO: Pretrained xlstm?
