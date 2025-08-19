import unittest

from GrooveModel.Tokenizer.MultiTaskDnaTokenizer import MultiTaskDnaTokenizer, MultiDnaToken
from GrooveModel.Utils.BeatUnit import decode_beat_unit
from GrooveModel.Utils.BeatsPerMinute import decode_bpm
from GrooveModel.Utils.DNAOffset import decode_offset_ticks


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

        tokens = MultiTaskDnaTokenizer.tokenize(song)
        self.assertEqual(len(tokens), 1)

        token = tokens[0]
        self.assertIsInstance(token, MultiDnaToken)
        self.assertGreater(token.Instrument, 0)
        self.assertGreater(token.Velocity, 0)
        self.assertEqual(decode_beat_unit(token.BeatUnit, absolute=False), 0)
        self.assertAlmostEqual(decode_offset_ticks(token.BeatUnitOffset, ticks_per_grid_unit=240, start_at_zero=True), 15, delta=1)
        self.assertAlmostEqual(decode_bpm(token.Bpm), 120, delta=4)

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
                    "Value": 3,
                    "IsEmpty": False,
                    "VelocityPerValuePart": {"0": 0.7, "1": 0.8},
                    "OffsetTicksPerValuePart": {"0": 5, "1": 10}
                }
            ]
        }

        tokens = MultiTaskDnaTokenizer.tokenize(song)
        self.assertEqual(len(tokens), 2)
        self.assertEqual(len(set(t.Instrument for t in tokens)), 2)

    def test_trim_leading_empty_measures(self):
        song = {
            "Bpm": 110,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 1,
            "DNAUnits": (
                    [{"Value": 0, "IsEmpty": True}] * 4 +
                    [{"Value": 1, "IsEmpty": False, "VelocityPerValuePart": {"1": 0.6},
                      "OffsetTicksPerValuePart": {"1": 0}}]
            )
        }

        tokens = MultiTaskDnaTokenizer.tokenize(song, trim_leading_empty_measures=True)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(decode_beat_unit(tokens[0].BeatUnit, absolute=False), 0)

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

        tokens = MultiTaskDnaTokenizer.tokenize(song, absolute_grid_units=True)
        self.assertEqual(len(tokens), 8)
        for i, token in enumerate(tokens):
            self.assertEqual(decode_beat_unit(token.BeatUnit, absolute=True), i)

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

        tokens = MultiTaskDnaTokenizer.tokenize(song)
        self.assertEqual(len(tokens), 3)

        for token in tokens:
            offset = decode_offset_ticks(token.BeatUnitOffset, ticks_per_grid_unit=120, start_at_zero=True)
            self.assertAlmostEqual(offset, 8, delta=1)

    def test_all_empty_units(self):
        song = {
            "Bpm": 100,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 1,
            "DNAUnits": [{"Value": 0, "IsEmpty": True} for _ in range(4)]
        }

        tokens = MultiTaskDnaTokenizer.tokenize(song)
        self.assertEqual(len(tokens), 0)

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

        tokens = MultiTaskDnaTokenizer.tokenize(song)
        self.assertEqual(len(tokens), 1)
        offset = decode_offset_ticks(tokens[0].BeatUnitOffset, ticks_per_grid_unit=960, start_at_zero=True)
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

        tokens = MultiTaskDnaTokenizer.tokenize(song)
        self.assertEqual(tokens, [])


if __name__ == '__main__':
    unittest.main()
