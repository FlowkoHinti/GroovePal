from typing import ClassVar

import torch
from pretty_midi import TimeSignature

from Configs import MAX_SEQUENCE_LENGTH
from GrooveModel.Tokenizer.Tokenizer import DnaTokenizer, SongData
from GrooveModel.Utils.BeatsPerMinute import encode_bpm
from GrooveModel.Utils.DNAGridFactor import GridFactors
from GrooveModel.Utils.DNAOffset import offset_to_percent_step
from GrooveModel.Utils.DNAValue import get_dna_instruments_list, encode_instrument, InstrumentValues
from GrooveModel.Utils.DNAVelocity import encode_velocity
from GrooveModel.Utils.TimeSignatures import encode_time_signature, TimeSignatures
from GrooveModel.Vocab import SequentialDnaVocab


class SequentialDnaTokenizer(DnaTokenizer):
    """
    Converts a SongData (parsed from JSON) into a sequence of vocab token IDs
    and aligned beat positions.

    Features:
      - Prepends meta tokens: BOS, BPM bin, time signature, grid factor.
      - Iterates over each grid unit in the song:
          * Inserts BAR token at bar starts.
          * If no instruments active → encodes Rest with vel=0, off=0.
          * Otherwise, encodes each active instrument with velocity and offset.
          * Appends a SEP token after each step.
      - Ends with EOS token.

    Returns:
      (torch.LongTensor token_ids, torch.LongTensor beat_positions)
    """

    vocab: ClassVar[SequentialDnaVocab] = SequentialDnaVocab()

    @staticmethod
    def tokenize(song_json,
                 trim_leading_empty_measures: bool = True,
                 absolute_grid_units: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        v = SequentialDnaTokenizer.vocab

        # Parse JSON into structured song data
        sd = SongData.from_json(song_json, trim_leading_empty_measures=trim_leading_empty_measures)

        # Encode BPM into a discrete bin index
        bpm_bin = encode_bpm(sd.bpm, include_padding=False)
        time_signature_name = encode_time_signature(sd.numerator, sd.denominator).name
        time_signature = TimeSignatures[time_signature_name]

        token_ids: list[int] = []
        beat_positions: list[int] = []
        ids_append, beats_append = token_ids.append, beat_positions.append
        ids_extend, beats_extend = token_ids.extend, beat_positions.extend

        # ---- META sequence ----
        ids_extend((
            v.ID_BOS,
            v.bpm_id(bpm_bin),
            v.time_sig_id(time_signature),
            v.grid_factor_id(GridFactors(sd.grid_factor)),
        ))
        beat_positions.extend([0] * len(token_ids))

        # Bar size in grid units
        grid_units_per_bar = sd.numerator * sd.grid_factor
        ticks_per_gu = sd.ticks_per_grid_unit

        if not sd.dna_units:
            raise ValueError("Song does not contain any DNA units")

        # ---- MAIN LOOP ----
        for grid_unit, dna_unit in enumerate(sd.dna_units):
            # Extract instruments (list of raw codes like 0,1,2,4,...)
            instruments = get_dna_instruments_list(dna_unit.get("Value", 0))
            velocities = dna_unit.get("VelocityPerValuePart", {})
            offsets = dna_unit.get("OffsetTicksPerValuePart", {})

            # Beat position for this step
            beat_unit = sd.beat_unit_for(grid_unit, absolute=absolute_grid_units)

            # Insert BAR marker at measure boundaries
            if grid_unit % grid_units_per_bar == 0:
                if MAX_SEQUENCE_LENGTH - len(token_ids) <= grid_units_per_bar * 3:
                    break
                ids_append(v.ID_BAR)
                beats_append(beat_unit)

            if not instruments:
                # No instruments → encode as REST
                rest_val = InstrumentValues.Rest
                ids_append(rest_val)
                beats_append(beat_unit)
                continue

            # Encode each active instrument
            for inst_val in instruments:
                key = str(inst_val)
                vel_idx = encode_velocity(velocities.get(key, 0), include_padding=False)
                off_step = offset_to_percent_step(offsets.get(key, 0), ticks_per_gu)

                ids_extend((v.instrument_id_from_value(inst_val),
                            v.vel_id(vel_idx),
                            v.off_id(off_step)))
                beats_extend((beat_unit, beat_unit, beat_unit))

            # Step separator
            ids_append(v.ID_SEP)
            beats_append(beat_unit)

        # ---- EOS ----
        last_beat = beat_positions[-1] if beat_positions else 0
        ids_append(v.ID_EOS)
        beats_append(last_beat)

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(beat_positions, dtype=torch.long)


# TODO: Pretrained xlstm?
