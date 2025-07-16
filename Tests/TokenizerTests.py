import unittest
from GrooveModel.Tokenizers import MultiDimDNATokenizer, DNAToken
from GrooveModel.Utils.DNAOffset import decode_offset_ticks


class TestMultiDimDNATokenizer(unittest.TestCase):

    def test_single_note_token(self):
        song = {
            "Bpm": 120,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 1,
            "NumberOfBars": 1,
            "DNA_ID": "TestSong01",
            "DNAUnits": [
                {
                    "Value": 1,  # Should produce instrument list
                    "IsEmpty": False,
                    "VelocityPerValuePart": {"1": 100},
                    "OffsetTicksPerValuePart": {"1": 15}
                }
            ]
        }

        tokens = MultiDimDNATokenizer.tokenize(song)
        self.assertEqual(len(tokens), 1)

        token = tokens[0]
        self.assertIsInstance(token, DNAToken)
        self.assertGreater(token.Instrument, 0)
        self.assertGreater(token.Velocity, 0)
        self.assertEqual(token.BeatUnit, 0)
        self.assertEqual(decode_offset_ticks(token.BeatUnitOffset, ticks_per_qn=song['TicksPerQuarterNote']), 15)
        self.assertEqual(token.Bpm, 120)

    def test_multiple_instruments(self):
        song = {
            "Bpm": 100,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 1,
            "NumberOfBars": 1,
            "DNA_ID": "MultiInstr",
            "DNAUnits": [
                {
                    "Value": 3,
                    "IsEmpty": False,
                    "VelocityPerValuePart": {"0": 80, "1": 90},
                    "OffsetTicksPerValuePart": {"0": 5, "1": 10}
                }
            ]
        }

        tokens = MultiDimDNATokenizer.tokenize(song)
        self.assertEqual(len(tokens), 2)
        instr_ids = sorted(t.Instrument for t in tokens)
        self.assertEqual(len(set(instr_ids)), 2)

    def test_empty_unit_fallback(self):
        song = {
            "Bpm": 90,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 1,
            "NumberOfBars": 1,
            "DNA_ID": "EmptyFallback",
            "DNAUnits": [
                {
                    "Value": 0,
                    "IsEmpty": True,
                    "VelocityPerValuePart": {},
                    "OffsetTicksPerValuePart": {}
                },
                {
                    "Value": 1,
                    "IsEmpty": False,
                    "VelocityPerValuePart": {"0": 80, "1": 90},
                    "OffsetTicksPerValuePart": {"0": 5, "1": 10}
                }
            ]
        }

        tokens = MultiDimDNATokenizer.tokenize(song)
        self.assertEqual(len(tokens), 2)

    def test_trim_leading_empty_measures(self):
        song = {
            "Bpm": 110,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 1,
            "NumberOfBars": 2,
            "DNA_ID": "TrimTest",
            "DNAUnits": (
                [{"Value": 0, "IsEmpty": True}] * 4 +  # First measure (4 units) – empty
                [{"Value": 1, "IsEmpty": False, "VelocityPerValuePart": {"1": 70}, "OffsetTicksPerValuePart": {"1": 0}}]
            )
        }

        tokens = MultiDimDNATokenizer.tokenize(song, trim_leading_empty_measures=True)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].BeatUnit, 0)  # Should reset after trimming

    def test_no_units(self):
        song = {
            "Bpm": 130,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 1,
            "NumberOfBars": 1,
            "DNA_ID": "NoUnits",
            "DNAUnits": []
        }

        tokens = MultiDimDNATokenizer.tokenize(song)
        self.assertEqual(tokens, [])

    def test_absolute_grid_units(self):
        song = {
            "Bpm": 110,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 2,
            "NumberOfBars": 1,
            "DNA_ID": "AbsGrid",
            "DNAUnits": [
                            {
                                "Value": 1,
                                "IsEmpty": False,
                                "VelocityPerValuePart": {"1": 90},
                                "OffsetTicksPerValuePart": {"1": 20}
                            }
                        ] * 8  # 2 measures worth of grid units
        }

        tokens = MultiDimDNATokenizer.tokenize(song, absolute_grid_units=True)
        self.assertEqual(len(tokens), 8)
        for i, token in enumerate(tokens):
            self.assertEqual(token.BeatUnit, i)

    def test_multiple_bars_and_grid_wrap(self):
        song = {
            "Bpm": 90,
            "Numerator": 3,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 2,
            "NumberOfBars": 2,
            "DNA_ID": "WrapGrid",
            "DNAUnits": [
                {
                    "Value": 1,
                    "IsEmpty": False,
                    "VelocityPerValuePart": {"1": 85},
                    "OffsetTicksPerValuePart": {"1": 0}
                }
                for _ in range(12)  # 2 bars * (3 beats * 2 grid units) = 12
            ]
        }

        tokens = MultiDimDNATokenizer.tokenize(song)
        self.assertEqual(len(tokens), 12)

        # BeatUnit should wrap every 6 (3 beats * 2 grid units)
        for i, token in enumerate(tokens):
            self.assertEqual(token.BeatUnit, i % 6)

    def test_incomplete_last_measure(self):
        song = {
            "Bpm": 105,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 1,
            "NumberOfBars": 1,
            "DNA_ID": "IncompleteBar",
            "DNAUnits": [
                {
                    "Value": 1,
                    "IsEmpty": False,
                    "VelocityPerValuePart": {"1": 60},
                    "OffsetTicksPerValuePart": {"1": 8}
                }
                for _ in range(3)  # Incomplete 4/4 bar
            ]
        }

        tokens = MultiDimDNATokenizer.tokenize(song)
        self.assertEqual(len(tokens), 3)

        for token in tokens:
            offset = decode_offset_ticks(token.BeatUnitOffset, ticks_per_qn=song["TicksPerQuarterNote"])
            self.assertEqual(offset, 8)

    def test_all_empty_units(self):
        song = {
            "Bpm": 100,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 480,
            "GridFactor": 1,
            "NumberOfBars": 1,
            "DNA_ID": "AllEmpty",
            "DNAUnits": [
                {"Value": 0, "IsEmpty": True} for _ in range(4)
            ]
        }

        tokens = MultiDimDNATokenizer.tokenize(song)
        self.assertEqual(len(tokens), 0)

    def test_offset_encoding_edge_case(self):
        song = {
            "Bpm": 120,
            "Numerator": 4,
            "Denominator": 4,
            "TicksPerQuarterNote": 960,  # higher resolution
            "GridFactor": 1,
            "NumberOfBars": 1,
            "DNA_ID": "HighTPQ",
            "DNAUnits": [
                {
                    "Value": 1,
                    "IsEmpty": False,
                    "VelocityPerValuePart": {"1": 127},
                    "OffsetTicksPerValuePart": {"1": 960}  # full quarter note offset
                }
            ]
        }

        tokens = MultiDimDNATokenizer.tokenize(song)
        self.assertEqual(len(tokens), 1)
        offset = decode_offset_ticks(tokens[0].BeatUnitOffset, ticks_per_qn=song["TicksPerQuarterNote"])
        self.assertEqual(offset, 960)


if __name__ == '__main__':
    unittest.main()