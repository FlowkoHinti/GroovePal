import unittest

from GrooveModel.Utils.BeatsPerMinute import encode_bpm, decode_bpm, BPM_RESOLUTION, BPM_TOKEN_SIZE
from GrooveModel.Utils.DNAGridFactor import GridFactors, RemappedGridFactors, encode_grid_factor, decode_grid_factor, \
    GRID_FACTOR_TOKEN_SIZE
from GrooveModel.Utils.DNAOffset import encode_offset_ticks, OFFSET_TOKEN_SIZE, decode_offset_ticks, \
    OFFSET_TICKS_RESOLUTION
from GrooveModel.Utils.DNAValue import InstrumentValues, RemappedInstrumentValues, get_dna_instruments_list, \
    encode_instrument, decode_instrument, dna_to_instruments_strings, instruments_strings_to_dna, DNA_VALUE_TOKEN_SIZE
from GrooveModel.Utils.DNAVelocity import VELOCITY_MIN, VELOCITY_MAX, VELOCITY_RESOLUTION, VELOCITY_TOKEN_SIZE, \
    encode_velocity, decode_velocity
from GrooveModel.Utils.TimeSignatures import TIME_SIGNATURE_TOKEN_SIZE, UNKNOWN_TIME_SIGNATURE_ID, TIME_SIGNATURE_RESOLUTION, \
    encode_time_signature, TIME_SIGNATURE_LOOKUP, decode_time_signature

# Mock SPECIAL_TOKEN_SIZE
SPECIAL_TOKEN_SIZE = 1

class TestDnaValueEncoding(unittest.TestCase):

    def test_instrument_enum_values(self):
        self.assertEqual(InstrumentValues.Kick, 1)
        self.assertEqual(InstrumentValues.Snare, 2)
        self.assertEqual(InstrumentValues.Toms, 4)
        self.assertEqual(InstrumentValues.HiHat, 8)
        self.assertEqual(InstrumentValues.Ride, 16)
        self.assertEqual(InstrumentValues.Crash, 32)

    def test_remapped_enum_auto_increment(self):
        self.assertEqual(RemappedInstrumentValues.Kick, SPECIAL_TOKEN_SIZE + 1)
        self.assertEqual(RemappedInstrumentValues.Snare, SPECIAL_TOKEN_SIZE + 2)
        self.assertEqual(RemappedInstrumentValues.Crash, SPECIAL_TOKEN_SIZE + 6)

    def test_get_dna_instruments_list(self):
        composite_value = InstrumentValues.Kick | InstrumentValues.Snare | InstrumentValues.HiHat
        result = get_dna_instruments_list(composite_value)
        expected = [InstrumentValues.Kick, InstrumentValues.Snare, InstrumentValues.HiHat]
        self.assertCountEqual(result, expected)

    def test_encode_instrument(self):
        self.assertEqual(encode_instrument(InstrumentValues.Toms), RemappedInstrumentValues.Toms)
        self.assertEqual(encode_instrument(InstrumentValues.Ride), RemappedInstrumentValues.Ride)

    def test_decode_instrument(self):
        self.assertEqual(decode_instrument(RemappedInstrumentValues.Snare), InstrumentValues.Snare)
        self.assertEqual(decode_instrument(RemappedInstrumentValues.Crash), InstrumentValues.Crash)

    def test_dna_to_instruments_strings(self):
        value = InstrumentValues.Kick | InstrumentValues.Crash
        result = dna_to_instruments_strings(value)
        self.assertCountEqual(result, ['Kick', 'Crash'])

    def test_instruments_strings_to_dna(self):
        instruments = ['Snare', 'HiHat']
        result = instruments_strings_to_dna(instruments)
        expected = InstrumentValues.Snare | InstrumentValues.HiHat
        self.assertEqual(result, expected)

    def test_round_trip_string_conversion(self):
        instruments = ['Kick', 'Ride', 'Toms']
        value = instruments_strings_to_dna(instruments)
        result = dna_to_instruments_strings(value)
        self.assertCountEqual(result, instruments)

    def test_dna_value_size(self):
        self.assertEqual(DNA_VALUE_TOKEN_SIZE, len(InstrumentValues) + SPECIAL_TOKEN_SIZE)


class TestGridFactorEncoding(unittest.TestCase):

    def test_grid_factor_enum_values(self):
        self.assertEqual(GridFactors.Quarter, 1)
        self.assertEqual(GridFactors.Eighth, 2)
        self.assertEqual(GridFactors.EighthTriplet, 3)
        self.assertEqual(GridFactors.Sixteenth, 4)
        self.assertEqual(GridFactors.SixteenthTriplet, 6)

    def test_remapped_enum_auto_increment(self):
        self.assertEqual(RemappedGridFactors.Quarter, SPECIAL_TOKEN_SIZE)
        self.assertEqual(RemappedGridFactors.Eighth, SPECIAL_TOKEN_SIZE + 1)
        self.assertEqual(RemappedGridFactors.SixteenthTriplet, SPECIAL_TOKEN_SIZE + 4)

    def test_encode_grid_factor(self):
        self.assertEqual(encode_grid_factor(GridFactors.Quarter), RemappedGridFactors.Quarter)
        self.assertEqual(encode_grid_factor(GridFactors.SixteenthTriplet), RemappedGridFactors.SixteenthTriplet)

    def test_decode_grid_factor(self):
        self.assertEqual(decode_grid_factor(RemappedGridFactors.EighthTriplet), GridFactors.EighthTriplet)
        self.assertEqual(decode_grid_factor(RemappedGridFactors.Sixteenth), GridFactors.Sixteenth)

    def test_round_trip_conversion(self):
        for factor in GridFactors:
            remapped = encode_grid_factor(factor)
            decoded = decode_grid_factor(remapped)
            self.assertEqual(decoded, factor)

    def test_grid_factors_size(self):
        self.assertEqual(GRID_FACTOR_TOKEN_SIZE, len(GridFactors) + SPECIAL_TOKEN_SIZE)


class TestOffsetEncoding(unittest.TestCase):

    def setUp(self):
        self.ticks_per_qn = 480  # standard MIDI resolution

    def test_offset_encoding_within_bounds(self):
        test_cases = [-480, -240, 0, 240, 480]
        for offset in test_cases:
            encoded = encode_offset_ticks(offset, self.ticks_per_qn)
            self.assertIsInstance(encoded, int)
            self.assertGreaterEqual(encoded, SPECIAL_TOKEN_SIZE)
            self.assertLess(encoded, OFFSET_TOKEN_SIZE)

    def test_offset_encoding_roundtrip(self):
        test_cases = [-480, -300, -1, 0, 1, 300, 480]
        for offset in test_cases:
            encoded = encode_offset_ticks(offset, self.ticks_per_qn)
            decoded = decode_offset_ticks(encoded, self.ticks_per_qn)
            self.assertAlmostEqual(decoded, offset, delta=1)  # Allow ±1 rounding error

    def test_offset_encoding_clamping(self):
        # Inputs outside the allowed tick range should be clamped
        too_small = encode_offset_ticks(-999, self.ticks_per_qn)
        too_large = encode_offset_ticks(999, self.ticks_per_qn)
        min_expected = encode_offset_ticks(-480, self.ticks_per_qn)
        max_expected = encode_offset_ticks(480, self.ticks_per_qn)
        self.assertEqual(too_small, min_expected)
        self.assertEqual(too_large, max_expected)

    def test_offset_encoding_start_at_zero_false(self):
        # Test with centering logic (not starting at zero)
        offset = 240
        encoded = encode_offset_ticks(offset, self.ticks_per_qn, start_at_zero=False)
        decoded = decode_offset_ticks(encoded, self.ticks_per_qn, start_at_zero=False)
        self.assertAlmostEqual(decoded, offset, delta=1)

    def test_offset_tokens_size(self):
        # Ensure total token count includes specials
        self.assertEqual(OFFSET_TOKEN_SIZE, OFFSET_TICKS_RESOLUTION + SPECIAL_TOKEN_SIZE)

class TestBPMEncoding(unittest.TestCase):

    def test_bpm_encode_decode_roundtrip(self):
        for bpm in [1, 60, 120, 180, 240, 300]:
            encoded = encode_bpm(bpm)
            decoded = decode_bpm(encoded)
            self.assertEqual(decoded, bpm, f"Round-trip failed for BPM: {bpm}")

    def test_encoded_bpm_has_special_token_offset(self):
        bpm = 100
        encoded = encode_bpm(bpm)
        self.assertEqual(encoded, bpm + SPECIAL_TOKEN_SIZE)

    def test_decoded_bpm_removes_offset(self):
        token = 150 + SPECIAL_TOKEN_SIZE
        decoded = decode_bpm(token)
        self.assertEqual(decoded, 150)

    def test_bpm_size_is_correct(self):
        self.assertEqual(BPM_TOKEN_SIZE, BPM_RESOLUTION + SPECIAL_TOKEN_SIZE)

    def test_encoding_bounds(self):
        encoded_min = encode_bpm(1)
        encoded_max = encode_bpm(BPM_RESOLUTION)
        self.assertEqual(decode_bpm(encoded_min), 1)
        self.assertEqual(decode_bpm(encoded_max), BPM_RESOLUTION)

        self.assertEqual(encoded_min, SPECIAL_TOKEN_SIZE + 1)
        self.assertEqual(encoded_max, BPM_TOKEN_SIZE)

class TestVelocityEncoding(unittest.TestCase):

    def test_velocity_constants(self):
        self.assertEqual(VELOCITY_MIN, 0)
        self.assertEqual(VELOCITY_MAX, 127)
        self.assertEqual(VELOCITY_RESOLUTION, 128)
        self.assertEqual(VELOCITY_TOKEN_SIZE, VELOCITY_RESOLUTION + SPECIAL_TOKEN_SIZE)

    def test_encoding_bounds(self):
        encoded_min = encode_velocity(0.0)
        encoded_max = encode_velocity(1.0)

        self.assertEqual(encoded_min, SPECIAL_TOKEN_SIZE)
        self.assertEqual(encoded_max, SPECIAL_TOKEN_SIZE + VELOCITY_RESOLUTION - 1)

    def test_decoding_bounds(self):
        decoded_min = decode_velocity(SPECIAL_TOKEN_SIZE)
        decoded_max = decode_velocity(SPECIAL_TOKEN_SIZE + VELOCITY_RESOLUTION - 1)

        self.assertAlmostEqual(decoded_min, 0.0, delta=1e-6)
        self.assertAlmostEqual(decoded_max, 1.0, delta=1e-6)

    def test_roundtrip_accuracy(self):
        test_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        for val in test_values:
            token = encode_velocity(val)
            decoded = decode_velocity(token)
            self.assertAlmostEqual(val, decoded, delta=1 / (VELOCITY_RESOLUTION - 1),
                                   msg=f"Failed roundtrip for velocity {val}")

    def test_encoding_precision_rounding(self):
        # Value slightly above halfway between two quantization levels should round up
        step_size = 1 / (VELOCITY_RESOLUTION - 1)
        midpoint = step_size * 5 + step_size / 2 + 1e-6  # halfway between step 5 and 6
        encoded = encode_velocity(midpoint)
        self.assertEqual(encoded, SPECIAL_TOKEN_SIZE + 6)


class TestTimeSignatureEncoding(unittest.TestCase):

    def test_constants(self):
        self.assertGreater(SPECIAL_TOKEN_SIZE, 0)
        self.assertEqual(TIME_SIGNATURE_TOKEN_SIZE, TIME_SIGNATURE_RESOLUTION + SPECIAL_TOKEN_SIZE)
        self.assertEqual(UNKNOWN_TIME_SIGNATURE_ID, TIME_SIGNATURE_TOKEN_SIZE)

    def test_encoding_valid_signatures(self):
        test_cases = {
            (4, 4): SPECIAL_TOKEN_SIZE,
            (3, 4): SPECIAL_TOKEN_SIZE + 1,
            (6, 8): SPECIAL_TOKEN_SIZE + 2,
            (3, 2): SPECIAL_TOKEN_SIZE + 11
        }
        for time_sig, expected_id in test_cases.items():
            encoded = encode_time_signature(*time_sig)
            self.assertEqual(encoded, expected_id)

    def test_decoding_valid_ids(self):
        for time_sig, encoded_id in TIME_SIGNATURE_LOOKUP.items():
            decoded = decode_time_signature(encoded_id)
            self.assertEqual(decoded, time_sig)

    def test_encode_unknown_signature_returns_fallback(self):
        self.assertEqual(
            encode_time_signature(5, 8),
            UNKNOWN_TIME_SIGNATURE_ID
        )
        self.assertEqual(
            encode_time_signature(11, 16),
            UNKNOWN_TIME_SIGNATURE_ID
        )

    def test_decode_unknown_id_raises(self):
        with self.assertRaises(ValueError):
            decode_time_signature(UNKNOWN_TIME_SIGNATURE_ID)

        with self.assertRaises(ValueError):
            decode_time_signature(9999)  # well out of range

    def test_denominator_validation(self):
        with self.assertRaises(ValueError):
            encode_time_signature(4, 1)  # invalid denominator

        with self.assertRaises(ValueError):
            encode_time_signature(4, 0)


if __name__ == '__main__':
    unittest.main()
