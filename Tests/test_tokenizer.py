import unittest
from itertools import chain

import torch

from GrooveModel.Tokenizer.MultiTaskDnaTokenizer import MultiTaskDnaTokenizer
from GrooveModel.Tokenizer.SequentialDnaTokenizer import SequentialDnaTokenizer
from GrooveModel.Utils.BeatUnit import decode_beat_unit
from GrooveModel.Utils.BeatsPerMinute import decode_bpm, encode_bpm, NUM_BPM_BINS
from GrooveModel.Utils.DNAOffset import decode_offset_ticks, offset_to_percent_step, OFFSET_STEPS
from GrooveModel.Utils.DNAValue import encode_instrument, InstrumentValues
from GrooveModel.Utils.DNAVelocity import decode_velocity, encode_velocity, SPECIAL_TOKEN_SIZE, \
    EFFECTIVE_VELOCITY_RESOLUTION
from GrooveModel.Vocab import SequentialDnaVocab


# ---------- shared dummy-data helpers (usable by both test classes) ----------
def make_step(
        value: int,
        *,
        vel: dict[int, float | int] | None = None,
        off: dict[int, int] | None = None
) -> dict:
    """
    Build a single DNA unit step.

    value: sparse/bitset instrument code (e.g., 0,1,2,4,8,...).
    vel: per-instrument velocity map; keys are raw instrument codes (ints).
    off: per-instrument offset ticks.
    is_empty: if provided, included as 'IsEmpty' (used by trim_empty_measures).
    """
    step = {"Value": value, "VelocityPerValuePart": {str(k): v for k, v in (vel or {}).items()},
            "OffsetTicksPerValuePart": {str(k): v for k, v in (off or {}).items()}, "IsEmpty": True if value == 0 else False}
    return step


def make_song(
        *,
        bpm: int = 120,
        numerator: int = 4,
        denominator: int = 4,
        grid_factor: int = 1,
        ticks_per_quarter: int | None = 480,
        ticks_per_grid_unit: int | None = 120,
        dna_id: str | None = None,
        steps: list[dict],
        include_trim_flag: bool = False,
) -> dict:
    """
    Build a minimal song JSON compatible with SongData.from_json.

    Notes:
    - Some producers set TicksPerQuarterNote; others set TicksPerGridUnit.
      SongData computes TicksPerGridUnit = TPQ // max(grid_factor,1) when absent.
    - Units field is 'DNAUnits' per your MultiTask tokenizer.
    """
    song = {
        "Bpm": bpm,
        "Numerator": numerator,
        "Denominator": denominator,
        "GridFactor": grid_factor,
        "DNAUnits": steps,
    }
    if ticks_per_quarter is not None:
        song["TicksPerQuarterNote"] = ticks_per_quarter
    if ticks_per_grid_unit is not None:
        song["TicksPerGridUnit"] = ticks_per_grid_unit
    if dna_id is not None:
        song["DNA_ID"] = dna_id
    if include_trim_flag:
        song["TrimLeadingEmptyMeasures"] = True
    return song


class TestMultiDimDNATokenizer(unittest.TestCase):

    def test_single_note_token(self):
        song = {
            "Bpm": 120,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 240,
            "GridFactor": 1,
            "DNA_ID": "TestSong01",
            "DNAUnits": [
                {
                    "Value": 1,
                    "IsEmpty": False,
                    "VelocityPerValuePart": {"1": 1},
                    "OffsetTicksPerValuePart": {"1": 15}
                }
            ]
        }

        tokens, beats = MultiTaskDnaTokenizer.tokenize(song)

        # Shapes
        self.assertIsInstance(tokens, torch.Tensor)
        self.assertIsInstance(beats, torch.Tensor)
        self.assertEqual(tokens.shape, (1, 7))
        self.assertEqual(beats.shape, (1,))

        t = tokens[0]

        # Instrument should not be REST
        self.assertNotEqual(int(t[0].item()), encode_instrument(InstrumentValues.Rest))

        # Velocity is encoded with SPECIAL_TOKEN_SIZE offset and decodes near 1.0
        self.assertGreaterEqual(int(t[1].item()), SPECIAL_TOKEN_SIZE)
        self.assertAlmostEqual(decode_velocity(int(t[1].item())), 1.0, delta=1e-6)

        # Beat unit and offset/bpm decoding
        self.assertEqual(decode_beat_unit(int(t[2].item()), absolute=False), 0)
        self.assertAlmostEqual(
            decode_offset_ticks(int(t[3].item()), ticks_per_grid_unit=240, start_at_zero=True),
            15, delta=1
        )
        self.assertAlmostEqual(decode_bpm(int(t[5].item())), 120, delta=4)

        # beat_positions should mirror the (possibly absolute) beat index per row
        self.assertEqual(int(beats[0].item()), 0)

    def test_multiple_instruments(self):
        song = {
            "Bpm": 100,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 1,
            "DNA_ID": "MultiInstr",
            "DNAUnits": [
                {
                    "Value": 3,  # two instruments set
                    "IsEmpty": False,
                    "VelocityPerValuePart": {"0": 0.7, "1": 0.8},
                    "OffsetTicksPerValuePart": {"0": 5, "1": 10}
                }
            ]
        }

        tokens, beats = MultiTaskDnaTokenizer.tokenize(song)
        self.assertEqual(tokens.shape[0], 2)
        # Two distinct instruments
        instruments = set(int(x) for x in tokens[:, 0].tolist())
        self.assertEqual(len(instruments), 2)
        # None should be REST
        self.assertNotIn(encode_instrument(InstrumentValues.Rest), instruments)
        # One beat position per emitted row
        self.assertEqual(beats.shape, (2,))

    def test_trim_leading_empty_measures(self):
        song = {
            "Bpm": 110,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 1,
            "DNAUnits": (
                    [{"Value": 0, "IsEmpty": True}] * 4 +  # a full empty bar
                    [{"Value": 1, "IsEmpty": False,
                      "VelocityPerValuePart": {"1": 0.6},
                      "OffsetTicksPerValuePart": {"1": 0}}]
            )
        }

        tokens, beats = MultiTaskDnaTokenizer.tokenize(song, trim_leading_empty_measures=True)
        # After trimming the leading empty bar, only the note remains
        self.assertEqual(tokens.shape, (1, 7))
        self.assertEqual(decode_beat_unit(int(tokens[0, 2].item()), absolute=False), 0)
        self.assertEqual(int(beats[0].item()), 0)

    def test_absolute_grid_units(self):
        song = {
            "Bpm": 110,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 2,
            "DNAUnits": [
                            {
                                "Value": 1,
                                "IsEmpty": False,
                                "VelocityPerValuePart": {"1": 0.8},
                                "OffsetTicksPerValuePart": {"1": 20}
                            }
                        ] * 8
        }

        tokens, beats = MultiTaskDnaTokenizer.tokenize(song, absolute_grid_units=True)
        self.assertEqual(tokens.shape[0], 8)
        # BeatUnit column should match absolute indices; beats tensor too
        for i in range(8):
            self.assertEqual(decode_beat_unit(int(tokens[i, 2].item()), absolute=True), i)
            self.assertEqual(int(beats[i].item()), i)

    def test_incomplete_last_measure(self):
        song = {
            "Bpm": 105,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerGridUnit": 120,
            "GridFactor": 1,
            "DNAUnits": [
                {
                    "Value": 1,
                    "IsEmpty": False,
                    "VelocityPerValuePart": {"1": 0.2},
                    "OffsetTicksPerValuePart": {"1": 8}
                }
                for _ in range(3)
            ]
        }

        tokens, beats = MultiTaskDnaTokenizer.tokenize(song)
        self.assertEqual(tokens.shape[0], 3)
        for i in range(3):
            offset = decode_offset_ticks(int(tokens[i, 3].item()), ticks_per_grid_unit=120, start_at_zero=True)
            self.assertAlmostEqual(offset, 8, delta=1)
            # Beats should be 0,1,2 in relative mode
            self.assertEqual(int(beats[i].item()), i)

    def test_all_empty_units(self):
        song = {
            "Bpm": 100,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 1,
            "DNAUnits": [{"Value": 0, "IsEmpty": True} for _ in range(4)]
        }

        tokens, beats = MultiTaskDnaTokenizer.tokenize(song)
        # Is empty
        self.assertEqual(tokens.shape, (0,))
        self.assertEqual(beats.shape, (0,))

    def test_offset_encoding_edge_case(self):
        song = {
            "Bpm": 120,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 960,
            "GridFactor": 1,
            "DNAUnits": [
                {
                    "Value": 1,
                    "IsEmpty": False,
                    "VelocityPerValuePart": {"1": 1},
                    "OffsetTicksPerValuePart": {"1": 480}
                }
            ]
        }

        tokens, beats = MultiTaskDnaTokenizer.tokenize(song)
        self.assertEqual(tokens.shape, (1, 7))
        offset = decode_offset_ticks(int(tokens[0, 3].item()), ticks_per_grid_unit=960, start_at_zero=True)
        self.assertEqual(offset, 480)

    def test_no_units(self):
        song = {
            "Bpm": 130,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 1,
            "DNAUnits": []
        }

        tokens, beats = MultiTaskDnaTokenizer.tokenize(song)
        # With no units, both outputs are empty tensors
        self.assertIsInstance(tokens, torch.Tensor)
        self.assertIsInstance(beats, torch.Tensor)
        self.assertEqual(tokens.numel(), 0)
        self.assertEqual(beats.numel(), 0)
        # At least confirm first dimension is zero (shape may be (0,) or (0,7) depending on PyTorch behavior)
        self.assertEqual(tokens.shape[0], 0)
        self.assertEqual(beats.shape[0], 0)


class TestSequentialTokenizer(unittest.TestCase):
    """
    Tests for SequentialDnaTokenizer + SequentialDnaVocab, aligned with SongData:
      - Uses DNAUnits / TicksPerGridUnit as per SongData.from_json
      - beat_unit_for(..., absolute) semantics
      - Does not rely on a time_signature enum (token may be 'Time_{num}_{den}')
    """

    def setUp(self):
        self.vocab = SequentialDnaVocab()

    def tokstr(self, ids: torch.Tensor) -> list[str]:
        return [self.vocab.token(i) for i in ids.tolist()]

    def test_beat_pos_len_equal_token_len(self):
        steps = [make_step(1) for _ in range(16)] + [make_step(0) for _ in range(12)] + [make_step(2) for _ in range(3)]
        song = make_song(steps=steps, numerator=4, denominator=4, grid_factor=4, ticks_per_grid_unit=120)

        ids, beats = SequentialDnaTokenizer.tokenize(song)
        self.assertEqual(ids.shape, beats.shape)


    def test_meta_bar_sep_eos(self):
        # 4/4, grid_factor=4 → 16 steps per bar (relative beats 0..15 then wrap)
        steps = [make_step(1) for _ in range(16)]
        song = make_song(steps=steps, numerator=4, denominator=4, grid_factor=4, ticks_per_grid_unit=120)

        ids, beats = SequentialDnaTokenizer.tokenize(song)
        toks = self.tokstr(ids)

        self.assertEqual(toks[0], "BOS")
        self.assertTrue(toks[1].startswith("Bpm_bin_"))
        # time token could be 'Time_{num}_{den}' or an enum; just ensure something occupies meta slot
        self.assertIn("BAR", toks)
        self.assertIn("SEP", toks)
        self.assertEqual(toks[-1], "EOS")
        self.assertEqual(ids.numel(), beats.numel())

    def test_two_instruments_same_step(self):
        val = InstrumentValues.Kick.value | InstrumentValues.Snare.value
        steps = [make_step(
            val,
            vel={InstrumentValues.Kick: 1, InstrumentValues.Snare: 0.6},
            off={InstrumentValues.Kick: 0, InstrumentValues.Snare: 0}
        )]
        song = make_song(steps=steps, grid_factor=4)

        ids, _ = SequentialDnaTokenizer.tokenize(song)
        toks = self.tokstr(ids)

        start = toks.index("BAR") + 1
        end = toks.index("SEP")
        step_tokens = toks[start:end]
        self.assertIn("Kick", step_tokens)
        self.assertIn("Snare", step_tokens)

    def test_trim_leading_empty_measures(self):
        # 1 empty bar at grid_factor=1 (4 steps), then one kick
        steps = [make_step(0) for _ in range(4)]
        steps += [make_step(InstrumentValues.Kick.value, vel={InstrumentValues.Kick.value: 0.6},
                            off={InstrumentValues.Kick.value: 0})]
        song = make_song(steps=steps, grid_factor=1)

        ids, beats = SequentialDnaTokenizer.tokenize(song, trim_leading_empty_measures=True)
        toks = self.tokstr(ids)

        # First real step follows BAR; ensure it's Kick and beat index is 0
        bar_idx = toks.index("BAR")
        self.assertEqual(toks[bar_idx + 1], "Kick")
        self.assertEqual(int(beats[bar_idx].item()), 0)

    def test_absolute_beats_progress(self):
        # 8 steps, grid_factor=2 → beats 0..7 with absolute=True
        steps = [make_step(InstrumentValues.Kick.value, vel={InstrumentValues.Kick.value: 0.8},
                           off={InstrumentValues.Kick.value: 20}) for _ in range(8)]
        song = make_song(steps=steps, grid_factor=2, ticks_per_grid_unit=120)

        ids, beats = SequentialDnaTokenizer.tokenize(song, absolute_grid_units=True)
        toks = self.tokstr(ids)

        # Collect beats tagged on SEP tokens
        sep_positions = [beats[i].item() for i, t in enumerate(toks) if t == "SEP"]
        self.assertEqual(sep_positions, list(range(8)))

    def test_velocity_offset_alignment(self):
        sn = InstrumentValues.Snare.value
        vel_raw = 73 * (1/128)
        off_ticks = 15
        steps = [make_step(sn, vel={sn: vel_raw}, off={sn: off_ticks})]
        song = make_song(steps=steps, ticks_per_grid_unit=120)

        ids, _ = SequentialDnaTokenizer.tokenize(song)
        toks = self.tokstr(ids)

        start = toks.index("BAR") + 1
        triplet = toks[start:start + 3]
        self.assertEqual(triplet[0], "Snare")

        vbin = encode_velocity(vel_raw, include_padding=False)
        self.assertEqual(triplet[1], f"Vel_{vbin}")

        obin = offset_to_percent_step(off_ticks, 120)
        obin_in = int(triplet[2].split("_")[-1])
        self.assertAlmostEqual(obin_in, obin, delta=1)

    def test_bpm_bin_range(self):
        for bpm in (40, 60, 90, 120, 200):
            steps = [make_step(1)]
            song = make_song(steps=steps, bpm=bpm)
            ids, _ = SequentialDnaTokenizer.tokenize(song)
            toks = self.tokstr(ids)
            bpm_tok = toks[1]  # BOS, BPM, ...
            self.assertTrue(bpm_tok.startswith("Bpm_bin_"))
            idx = int(bpm_tok.split("_")[-1])
            self.assertEqual(idx, encode_bpm(bpm, include_padding=False))
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, NUM_BPM_BINS)

    def test_empty_units(self):
        steps = [make_step(0) for _ in range(4)]
        song = make_song(steps=steps, grid_factor=1, ticks_per_grid_unit=120)

        with self.assertRaises(ValueError):
            ids, _ = SequentialDnaTokenizer.tokenize(song)

    def test_vel_off_bounds(self):
        steps = [
            make_step(InstrumentValues.Kick.value, vel={InstrumentValues.Kick.value: v},
                      off={InstrumentValues.Kick.value: o})
            for v, o in ((0, -60), (0.5, 0), (1, 60))
        ]
        song = make_song(steps=steps, ticks_per_grid_unit=120)
        ids, _ = SequentialDnaTokenizer.tokenize(song)
        v = self.vocab
        toks = self.tokstr(ids)
        vel_tokens = [t for t in toks if t.startswith("Vel_")]
        off_tokens = [t for t in toks if t.startswith("Off_Step_")]
        for t in vel_tokens:
            binv = int(t.split("_")[-1])
            self.assertGreaterEqual(binv, 0)
            self.assertLess(binv, EFFECTIVE_VELOCITY_RESOLUTION)
        for t in off_tokens:
            bino = int(t.split("_")[-1])
            self.assertGreaterEqual(bino, -OFFSET_STEPS)
            self.assertLessEqual(bino, OFFSET_STEPS)


if __name__ == '__main__':
    unittest.main()
