import unittest
import torch

from GrooveModel.Tokenizer.MultiTaskDnaTokenizer import MultiTaskDnaTokenizer
from GrooveModel.Utils.BeatUnit import decode_beat_unit
from GrooveModel.Utils.BeatsPerMinute import decode_bpm
from GrooveModel.Utils.DNAOffset import decode_offset_ticks
from GrooveModel.Utils.DNAValue import encode_instrument, InstrumentValues
from GrooveModel.Utils.DNAVelocity import decode_velocity, SPECIAL_TOKEN_SIZE


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


if __name__ == '__main__':
    unittest.main()
