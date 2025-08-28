from dataclasses import dataclass

import torch

from Configs import MAX_SEQUENCE_LENGTH
from GrooveModel.Tokenizer.Tokenizer import DnaTokenizer, SongData
from GrooveModel.Utils.BeatUnit import encode_beat_unit
from GrooveModel.Utils.BeatsPerMinute import encode_bpm
from GrooveModel.Utils.DNAGridFactor import encode_grid_factor
from GrooveModel.Utils.DNAOffset import encode_offset_ticks
from GrooveModel.Utils.DNAValue import get_dna_instruments_list, encode_instrument, InstrumentValues
from GrooveModel.Utils.DNAVelocity import encode_velocity
from GrooveModel.Utils.TimeSignatures import encode_time_signature



# TODO: MAKE MORE PERFORMANT (DO ALL IN TENSOR SPACE ALREADY) -> OR Pretokenize

class MultiTaskDnaTokenizer(DnaTokenizer):

    @staticmethod
    def tokenize(song_json: dict,
                 trim_leading_empty_measures: bool = True,
                 absolute_grid_units: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes a song dictionary into a tensor.

        Returns:
          tokens: (N, 7) long tensor (your encoded rows)
          beat_positions: (N,) long tensor with the beat_position
        """
        song_data = SongData.from_json(song_json, trim_leading_empty_measures=trim_leading_empty_measures)

        time_signature_encoded = encode_time_signature(song_data.numerator, song_data.denominator)
        grid_factor_encoded = encode_grid_factor(song_data.grid_factor)
        bpm_encoded = encode_bpm(song_data.bpm)

        grid_units_per_bar = song_data.numerator * song_data.grid_factor

        rows = []
        beat_positions = []

        for grid_unit, dna_unit in enumerate(song_data.dna_units):
            instruments = get_dna_instruments_list(dna_unit.get("Value", 0))
            velocities = dna_unit.get("VelocityPerValuePart", {}) or {}
            offsets = dna_unit.get("OffsetTicksPerValuePart", {}) or {}

            beat_unit = song_data.beat_unit_for(grid_unit, absolute=absolute_grid_units)
            encoded_beat_unit = encode_beat_unit(beat_unit, absolute=absolute_grid_units)

            if grid_unit % grid_units_per_bar == 0:
                if MAX_SEQUENCE_LENGTH - len(rows) <= grid_units_per_bar * 3:
                    break
            # Rest
            if not instruments:
                rows.append([
                    encode_instrument(InstrumentValues.Rest),
                    encode_velocity(0),
                    encoded_beat_unit,
                    encode_offset_ticks(0, song_data.ticks_per_grid_unit),
                    grid_factor_encoded,
                    bpm_encoded,
                    time_signature_encoded
                ])
                beat_positions.append(beat_unit)
            else:
                # Entry for every instrument at beat unit
                for instrument in instruments:
                    rows.append([
                        encode_instrument(instrument),
                        encode_velocity(velocities.get(str(instrument), 0)),
                        encoded_beat_unit,
                        encode_offset_ticks(offsets.get(str(instrument), 0),
                                            song_data.ticks_per_grid_unit,
                                            start_at_zero=True),
                        grid_factor_encoded,
                        bpm_encoded,
                        time_signature_encoded
                    ])
                    beat_positions.append(beat_unit)

        tokens = torch.tensor(rows, dtype=torch.long)
        beat_pos_tensor = torch.tensor(beat_positions, dtype=torch.long)
        return tokens, beat_pos_tensor
