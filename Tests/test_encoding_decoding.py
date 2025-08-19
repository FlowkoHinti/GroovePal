import unittest

from GrooveModel.Utils.BeatUnit import BEAT_UNIT_TOKEN_SIZE_RELATIVE, MAX_GRID_UNITS_PER_BAR, MAX_GRID_UNITS_PER_SONG, \
    BEAT_UNIT_TOKEN_SIZE_ABSOLUTE, encode_beat_unit, decode_beat_unit
from GrooveModel.Utils.BeatsPerMinute import encode_bpm, decode_bpm, BPM_TOKEN_SIZE, MIN_BPM, MAX_BPM, \
    NUM_BPM_BINS
from GrooveModel.Utils.DNAGridFactor import GridFactors, RemappedGridFactors, encode_grid_factor, decode_grid_factor, \
    GRID_FACTOR_TOKEN_SIZE
from GrooveModel.Utils.DNAOffset import encode_offset_ticks, OFFSET_TOKEN_SIZE, decode_offset_ticks, \
    OFFSET_TICKS_RESOLUTION, normalize_offset, denormalize_offset
from GrooveModel.Utils.DNAValue import InstrumentValues, RemappedInstrumentValues, get_dna_instruments_list, \
    encode_instrument, decode_instrument, dna_to_instruments_strings, instruments_strings_to_dna, DNA_VALUE_TOKEN_SIZE
from GrooveModel.Utils.DNAVelocity import VELOCITY_MIN, VELOCITY_MAX, VELOCITY_RESOLUTION, VELOCITY_TOKEN_SIZE, \
    encode_velocity, decode_velocity
from GrooveModel.Utils.TimeSignatures import TIME_SIGNATURE_TOKEN_SIZE, UNKNOWN_TIME_SIGNATURE_ID, \
    encode_time_signature, decode_time_signature, TimeSignatures, RemappedTimeSignatures

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
        self.assertEqual(GridFactors.Quarter_Grid, 1)
        self.assertEqual(GridFactors.Eighth_Grid, 2)
        self.assertEqual(GridFactors.EighthTriplet_Grid, 3)
        self.assertEqual(GridFactors.Sixteenth_Grid, 4)
        self.assertEqual(GridFactors.SixteenthTriplet_Grid, 6)

    def test_remapped_enum_auto_increment(self):
        self.assertEqual(RemappedGridFactors.Quarter_Grid, SPECIAL_TOKEN_SIZE)
        self.assertEqual(RemappedGridFactors.Eighth_Grid, SPECIAL_TOKEN_SIZE + 1)
        self.assertEqual(RemappedGridFactors.SixteenthTriplet_Grid, SPECIAL_TOKEN_SIZE + 4)

    def test_encode_grid_factor(self):
        self.assertEqual(encode_grid_factor(GridFactors.Quarter_Grid), RemappedGridFactors.Quarter_Grid)
        self.assertEqual(encode_grid_factor(GridFactors.SixteenthTriplet_Grid),
                         RemappedGridFactors.SixteenthTriplet_Grid)

    def test_decode_grid_factor(self):
        self.assertEqual(decode_grid_factor(RemappedGridFactors.EighthTriplet_Grid), GridFactors.EighthTriplet_Grid)
        self.assertEqual(decode_grid_factor(RemappedGridFactors.Sixteenth_Grid), GridFactors.Sixteenth_Grid)

    def test_round_trip_conversion(self):
        for factor in GridFactors:
            remapped = encode_grid_factor(factor)
            decoded = decode_grid_factor(remapped)
            self.assertEqual(decoded, factor)

    def test_grid_factors_size(self):
        self.assertEqual(GRID_FACTOR_TOKEN_SIZE, len(GridFactors) + SPECIAL_TOKEN_SIZE)


class TestOffsetEncoding(unittest.TestCase):

    def setUp(self):
        self.ticks_per_grid_unit = 120
        self.max_half = self.ticks_per_grid_unit // 2  # 60

    def test_offset_encoding_within_bounds(self):
        test_cases = [-self.max_half, -30, 0, 30, self.max_half]  # [-60, -30, 0, 30, 60]
        for offset in test_cases:
            encoded = encode_offset_ticks(offset, self.ticks_per_grid_unit, start_at_zero=True)
            self.assertIsInstance(encoded, int)
            self.assertGreaterEqual(encoded, SPECIAL_TOKEN_SIZE)
            self.assertLess(encoded, OFFSET_TOKEN_SIZE)

    def test_offset_encoding_roundtrip(self):
        test_cases = [-self.max_half, -45, -1, 0, 1, 45, self.max_half]  # within ±60
        for offset in test_cases:
            encoded = encode_offset_ticks(offset, self.ticks_per_grid_unit)
            decoded = decode_offset_ticks(encoded, self.ticks_per_grid_unit)
            self.assertAlmostEqual(decoded, offset, delta=1)  # Allow ±1 tick rounding error

    def test_offset_encoding_clamping(self):
        # Values beyond ±ticks_per_grid_unit/2 should be clamped to ±max_half
        too_small = encode_offset_ticks(-999, self.ticks_per_grid_unit)
        too_large = encode_offset_ticks(999, self.ticks_per_grid_unit)
        min_expected = encode_offset_ticks(-self.max_half, self.ticks_per_grid_unit)
        max_expected = encode_offset_ticks(self.max_half, self.ticks_per_grid_unit)
        self.assertEqual(too_small, min_expected)
        self.assertEqual(too_large, max_expected)

    def test_offset_encoding_start_at_zero_false(self):
        # When start_at_zero is False, center the range around 0 (still within ±max_half)
        offset = 30
        encoded = encode_offset_ticks(offset, self.ticks_per_grid_unit, start_at_zero=False)
        decoded = decode_offset_ticks(encoded, self.ticks_per_grid_unit, start_at_zero=False)
        self.assertAlmostEqual(decoded, offset, delta=1)

    def test_offset_tokens_size(self):
        # Confirm size of total token space includes special tokens
        self.assertEqual(OFFSET_TOKEN_SIZE, OFFSET_TICKS_RESOLUTION + SPECIAL_TOKEN_SIZE)

    # --- Normalization range & endpoints ---

    def test_normalize_start_at_zero_range_and_endpoints(self):
        # [-max_half, 0, +max_half] -> [0.0, 0.5, 1.0]
        self.assertAlmostEqual(
            normalize_offset(-self.max_half, self.ticks_per_grid_unit, start_at_zero=True), 0.0, places=7
        )
        self.assertAlmostEqual(
            normalize_offset(0, self.ticks_per_grid_unit, start_at_zero=True), 0.5, places=7
        )
        self.assertAlmostEqual(
            normalize_offset(self.max_half, self.ticks_per_grid_unit, start_at_zero=True), 1.0, places=7
        )

    def test_normalize_centered_range_and_endpoints(self):
        # [-max_half, 0, +max_half] -> [-1.0, 0.0, 1.0]
        self.assertAlmostEqual(
            normalize_offset(-self.max_half, self.ticks_per_grid_unit, start_at_zero=False), -1.0, places=7
        )
        self.assertAlmostEqual(
            normalize_offset(0, self.ticks_per_grid_unit, start_at_zero=False), 0.0, places=7
        )
        self.assertAlmostEqual(
            normalize_offset(self.max_half, self.ticks_per_grid_unit, start_at_zero=False), 1.0, places=7
        )

    # --- Round-trip fidelity ---

    def test_normalize_roundtrip_start_at_zero(self):
        test_cases = [-self.max_half, -45, -1, 0, 1, 45, self.max_half]
        for offset in test_cases:
            n = normalize_offset(offset, self.ticks_per_grid_unit, start_at_zero=True)
            recovered = denormalize_offset(n, self.ticks_per_grid_unit, start_at_zero=True)
            self.assertAlmostEqual(recovered, offset, delta=1)  # allow ±1 tick

    def test_normalize_roundtrip_centered(self):
        test_cases = [-self.max_half, -45, -1, 0, 1, 45, self.max_half]
        for offset in test_cases:
            n = normalize_offset(offset, self.ticks_per_grid_unit, start_at_zero=False)
            recovered = denormalize_offset(n, self.ticks_per_grid_unit, start_at_zero=False)
            self.assertAlmostEqual(recovered, offset, delta=1)  # allow ±1 tick

    # --- Clamping behavior on input offsets ---

    def test_normalize_clamps_input_start_at_zero(self):
        # Inputs outside ±max_half should clamp to endpoints 0.0 and 1.0
        too_small = normalize_offset(-999, self.ticks_per_grid_unit, start_at_zero=True)
        too_large = normalize_offset(999, self.ticks_per_grid_unit, start_at_zero=True)
        self.assertEqual(too_small, 0.0)
        self.assertEqual(too_large, 1.0)

    def test_normalize_clamps_input_centered(self):
        # Inputs outside ±max_half should clamp to endpoints -1.0 and 1.0
        too_small = normalize_offset(-999, self.ticks_per_grid_unit, start_at_zero=False)
        too_large = normalize_offset(999, self.ticks_per_grid_unit, start_at_zero=False)
        self.assertEqual(too_small, -1.0)
        self.assertEqual(too_large, 1.0)

    # --- Denormalization correctness on canonical values ---

    def test_denormalize_from_canonical_values_start_at_zero(self):
        self.assertEqual(
            denormalize_offset(0.0, self.ticks_per_grid_unit, start_at_zero=True),
            -self.max_half
        )
        self.assertEqual(
            denormalize_offset(0.5, self.ticks_per_grid_unit, start_at_zero=True),
            0
        )
        self.assertEqual(
            denormalize_offset(1.0, self.ticks_per_grid_unit, start_at_zero=True),
            self.max_half
        )

    def test_denormalize_from_canonical_values_centered(self):
        self.assertEqual(
            denormalize_offset(-1.0, self.ticks_per_grid_unit, start_at_zero=False),
            -self.max_half
        )
        self.assertEqual(
            denormalize_offset(0.0, self.ticks_per_grid_unit, start_at_zero=False),
            0
        )
        self.assertEqual(
            denormalize_offset(1.0, self.ticks_per_grid_unit, start_at_zero=False),
            self.max_half
        )

    # --- Type checks ---

    def test_normalize_and_denormalize_types(self):
        n = normalize_offset(0, self.ticks_per_grid_unit, start_at_zero=True)
        self.assertIsInstance(n, float)
        d = denormalize_offset(0.5, self.ticks_per_grid_unit, start_at_zero=True)
        self.assertIsInstance(d, int)


class TestBPMEncoding(unittest.TestCase):

    def setUp(self):
        # Derived values used in multiple tests
        self.range_ = MAX_BPM - MIN_BPM
        self.bin_width = self.range_ / NUM_BPM_BINS
        self.eps = self.bin_width * 1e-6  # tiny nudge to avoid boundary ambiguity

    # --- Helpers ---
    def expected_bin_index(self, bpm: float) -> int:
        rel = (bpm - MIN_BPM) / self.range_
        idx = int(rel * NUM_BPM_BINS)
        return min(max(idx, 0), NUM_BPM_BINS - 1)

    def expected_midpoint(self, idx: int) -> float:
        return MIN_BPM + (idx + 0.5) * self.bin_width

    # --- Core behavior ---

    def test_encode_maps_to_correct_bins(self):
        """Encoding places BPMs into the expected bin indices."""
        # Choose BPMs firmly inside a few bins: 0, middle, last
        candidates = [
            MIN_BPM + self.eps,  # inside bin 0
            MIN_BPM + (NUM_BPM_BINS // 2) * self.bin_width + self.bin_width * 0.25,
            MAX_BPM - self.bin_width * 0.75,  # near the end, but < MAX_BPM
        ]
        for bpm in candidates:
            idx = self.expected_bin_index(bpm)
            token = encode_bpm(bpm, include_padding=True)
            self.assertEqual(token, SPECIAL_TOKEN_SIZE + idx)

            token_np = encode_bpm(bpm, include_padding=False)
            self.assertEqual(token_np, idx)

    def test_decode_returns_bin_midpoint_float(self):
        """Decoding returns the exact bin midpoint when as_int=False."""
        for idx in [0, NUM_BPM_BINS // 3, NUM_BPM_BINS // 2, NUM_BPM_BINS - 1]:
            token = SPECIAL_TOKEN_SIZE + idx
            decoded = decode_bpm(token, include_padding=True, as_int=False)
            exp = self.expected_midpoint(idx)
            self.assertAlmostEqual(decoded, exp, places=9)

            # no-padding variant
            token_np = idx
            decoded_np = decode_bpm(token_np, include_padding=False, as_int=False)
            self.assertAlmostEqual(decoded_np, exp, places=9)

    def test_roundtrip_yields_bin_midpoint(self):
        """Encode→decode yields the midpoint of the selected bin (not the original BPM)."""
        test_bpms = [
            MIN_BPM,
            60,
            120,
            180,
            240,
            MAX_BPM - 1,  # still inside range per spec (MAX_BPM is exclusive)
            MIN_BPM + self.bin_width * 0.49,  # near edge but inside bin 0
            MIN_BPM + self.bin_width * 1.01,  # crosses into bin 1
        ]
        for bpm in test_bpms:
            idx = self.expected_bin_index(bpm)
            token = encode_bpm(bpm)
            decoded = decode_bpm(token, as_int=False)  # check exact midpoint
            exp = self.expected_midpoint(idx)
            self.assertAlmostEqual(decoded, exp, places=9, msg=f"Failed at BPM={bpm}")

    def test_decoding_as_int_rounds_midpoint(self):
        """When as_int=True, midpoint is rounded to nearest int."""
        # Pick a known bin and verify rounding behavior
        idx = NUM_BPM_BINS // 2
        token = SPECIAL_TOKEN_SIZE + idx
        mid = self.expected_midpoint(idx)
        decoded_int = decode_bpm(token, as_int=True)
        self.assertEqual(decoded_int, int(round(mid)))

    def test_bpm_token_size_is_correct(self):
        self.assertEqual(BPM_TOKEN_SIZE, NUM_BPM_BINS + SPECIAL_TOKEN_SIZE)

    def test_encoding_bounds(self):
        """MIN_BPM maps to bin 0; values very close to MAX_BPM map to last bin."""
        # MIN_BPM
        t_min = encode_bpm(MIN_BPM)
        self.assertEqual(t_min, SPECIAL_TOKEN_SIZE + 0)
        d_min = decode_bpm(t_min, as_int=False)
        self.assertAlmostEqual(d_min, self.expected_midpoint(0), places=9)

        # A value guaranteed to lie in the last bin (but still < MAX_BPM)
        bpm_last_bin = MAX_BPM - self.eps
        t_last = encode_bpm(bpm_last_bin)
        self.assertEqual(t_last, SPECIAL_TOKEN_SIZE + (NUM_BPM_BINS - 1))
        d_last = decode_bpm(t_last, as_int=False)
        self.assertAlmostEqual(d_last, self.expected_midpoint(NUM_BPM_BINS - 1), places=9)

    # --- Error cases ---

    def test_out_of_range_bpm_raises_error(self):
        with self.assertRaises(ValueError):
            encode_bpm(MIN_BPM - 1)
        with self.assertRaises(ValueError):
            encode_bpm(MAX_BPM)  # MAX_BPM is exclusive

    def test_out_of_range_token_raises_error(self):
        # include_padding=True invalid tokens
        with self.assertRaises(ValueError):
            decode_bpm(SPECIAL_TOKEN_SIZE - 1)  # below padded range
        with self.assertRaises(ValueError):
            decode_bpm(SPECIAL_TOKEN_SIZE + NUM_BPM_BINS)  # one above max padded token

        # include_padding=False invalid indices
        with self.assertRaises(ValueError):
            decode_bpm(-1, include_padding=False)
        with self.assertRaises(ValueError):
            decode_bpm(NUM_BPM_BINS, include_padding=False)

    def test_no_padding_consistency(self):
        """Encoding/decoding with include_padding=False is consistent and matches padded results after adjusting offset."""
        bpm = 137.3
        idx = self.expected_bin_index(bpm)

        tok_padded = encode_bpm(bpm, include_padding=True)
        tok_np = encode_bpm(bpm, include_padding=False)
        self.assertEqual(tok_padded, SPECIAL_TOKEN_SIZE + idx)
        self.assertEqual(tok_np, idx)

        d_padded = decode_bpm(tok_padded, include_padding=True, as_int=False)
        d_np = decode_bpm(tok_np, include_padding=False, as_int=False)
        self.assertAlmostEqual(d_padded, d_np, places=9)


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
        self.assertEqual(TIME_SIGNATURE_TOKEN_SIZE, len(TimeSignatures) + SPECIAL_TOKEN_SIZE)
        self.assertEqual(UNKNOWN_TIME_SIGNATURE_ID, RemappedTimeSignatures.Unknown)

    def test_encoding_valid_signatures(self):
        test_cases = {
            (4, 4): RemappedTimeSignatures.Time_4_4,
            (3, 4): RemappedTimeSignatures.Time_3_4,
            (6, 8): RemappedTimeSignatures.Time_6_8,
            (3, 2): RemappedTimeSignatures.Time_3_2,
        }
        for time_sig, expected_id in test_cases.items():
            encoded = encode_time_signature(*time_sig)
            self.assertEqual(encoded, expected_id)

    def test_decoding_valid_ids(self):
        test_cases = {
            RemappedTimeSignatures.Time_4_4: (4, 4),
            RemappedTimeSignatures.Time_3_4: (3, 4),
            RemappedTimeSignatures.Time_6_8: (6, 8),
            RemappedTimeSignatures.Time_3_2: (3, 2),
        }
        for encoded_id, expected_time_sig in test_cases.items():
            decoded = decode_time_signature(encoded_id)
            self.assertEqual(decoded, expected_time_sig)

    def test_encode_unknown_signature_returns_fallback(self):
        self.assertEqual(
            encode_time_signature(5, 8),
            RemappedTimeSignatures.Unknown
        )
        self.assertEqual(
            encode_time_signature(11, 16),
            RemappedTimeSignatures.Unknown
        )

    def test_decode_unknown_id_returns_placeholder(self):
        self.assertEqual(
            decode_time_signature(RemappedTimeSignatures.Unknown),
            ("?", "?")
        )
        self.assertEqual(
            decode_time_signature(9999),
            ("?", "?")
        )

    def test_denominator_validation(self):
        with self.assertRaises(ValueError):
            encode_time_signature(4, 1)

        with self.assertRaises(ValueError):
            encode_time_signature(4, 0)


class TestBeatUnitEncoding(unittest.TestCase):

    def test_constants(self):
        self.assertEqual(BEAT_UNIT_TOKEN_SIZE_RELATIVE, MAX_GRID_UNITS_PER_BAR + SPECIAL_TOKEN_SIZE)
        self.assertEqual(BEAT_UNIT_TOKEN_SIZE_ABSOLUTE, MAX_GRID_UNITS_PER_SONG + SPECIAL_TOKEN_SIZE)

    def test_encode_decode_relative(self):
        for pos in [0, 1, 35, MAX_GRID_UNITS_PER_BAR - 1]:
            token = encode_beat_unit(pos, absolute=False)
            decoded = decode_beat_unit(token, absolute=False)
            self.assertEqual(decoded, pos)

    def test_encode_decode_absolute(self):
        for pos in [0, 10, 127, MAX_GRID_UNITS_PER_SONG - 1]:
            token = encode_beat_unit(pos, absolute=True)
            decoded = decode_beat_unit(token, absolute=True)
            self.assertEqual(decoded, pos)

    def test_relative_token_range(self):
        token = encode_beat_unit(0, absolute=False)
        self.assertEqual(token, SPECIAL_TOKEN_SIZE)
        token = encode_beat_unit(MAX_GRID_UNITS_PER_BAR - 1, absolute=False)
        self.assertEqual(token, SPECIAL_TOKEN_SIZE + MAX_GRID_UNITS_PER_BAR - 1)

    def test_absolute_token_range(self):
        token = encode_beat_unit(0, absolute=True)
        self.assertEqual(token, SPECIAL_TOKEN_SIZE)
        token = encode_beat_unit(MAX_GRID_UNITS_PER_SONG - 1, absolute=True)
        self.assertEqual(token, SPECIAL_TOKEN_SIZE + MAX_GRID_UNITS_PER_SONG - 1)

    def test_out_of_bounds_relative(self):
        with self.assertRaises(ValueError):
            encode_beat_unit(MAX_GRID_UNITS_PER_BAR + 1, absolute=False)
        with self.assertRaises(ValueError):
            decode_beat_unit(SPECIAL_TOKEN_SIZE + MAX_GRID_UNITS_PER_BAR + 1, absolute=False)

    def test_out_of_bounds_absolute(self):
        with self.assertRaises(ValueError):
            encode_beat_unit(MAX_GRID_UNITS_PER_SONG + 1, absolute=True)
        with self.assertRaises(ValueError):
            decode_beat_unit(SPECIAL_TOKEN_SIZE + MAX_GRID_UNITS_PER_SONG + 1, absolute=True)


if __name__ == '__main__':
    unittest.main()
