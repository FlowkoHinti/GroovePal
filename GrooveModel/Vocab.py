# Token defines
# also mask tokens/ start tokens / end tokens
from dataclasses import dataclass, field
from itertools import chain
from typing import ClassVar, Mapping, Sequence, Counter

from GrooveModel.Utils.BeatUnit import BEAT_UNIT_TOKEN_SIZE_ABSOLUTE, BEAT_UNIT_TOKEN_SIZE_RELATIVE
from GrooveModel.Utils.BeatsPerMinute import BPM_TOKEN_SIZE, NUM_BPM_BINS
from GrooveModel.Utils.DNAGridFactor import GRID_FACTOR_TOKEN_SIZE, GridFactors
from GrooveModel.Utils.DNAOffset import OFFSET_TOKEN_SIZE, OFFSET_TICKS_RESOLUTION, OFFSET_STEPS
from GrooveModel.Utils.DNAValue import DNA_VALUE_TOKEN_SIZE, InstrumentValues
from GrooveModel.Utils.DNAVelocity import VELOCITY_TOKEN_SIZE, VELOCITY_RESOLUTION, EFFECTIVE_VELOCITY_RESOLUTION
from GrooveModel.Utils.TimeSignatures import TIME_SIGNATURE_TOKEN_SIZE, TimeSignatures

# --- MULTI TAKS VOCAB ---
# vocab sizes (+1 for padding token)
INSTRUMENT_VOCAB_SIZE = DNA_VALUE_TOKEN_SIZE
VELOCITY_VOCAB_SIZE = VELOCITY_TOKEN_SIZE
OFFSET_VOCAB_SIZE = OFFSET_TOKEN_SIZE
BPM_VOCAB_SIZE = BPM_TOKEN_SIZE
TIME_SIGNATURE_VOCAB_SIZE = TIME_SIGNATURE_TOKEN_SIZE
GRID_FACTOR_VOCAB_SIZE = GRID_FACTOR_TOKEN_SIZE
BEAT_UNIT_ABSOLUTE_VOCAB_SIZE = BEAT_UNIT_TOKEN_SIZE_ABSOLUTE
BEAT_UNIT_RELATIVE_VOCAB_SIZE = BEAT_UNIT_TOKEN_SIZE_RELATIVE


# --- SEQUENTIAL VOCAB ---
@dataclass(slots=True, frozen=True)
class SequentialDnaVocab:
    """
    Vocabulary for sequential DNA-style music tokenization.
    """

    # Immutable token groups
    SPECIAL_TOKENS: ClassVar[tuple[str, ...]] = ("PAD", "BOS", "EOS", "MASK", "SEP", "BAR", "CLS")
    INSTRUMENTS: ClassVar[tuple[str, ...]] = tuple(i.name for i in InstrumentValues)
    GRID_FACTORS: ClassVar[tuple[str, ...]] = tuple(gf.name for gf in GridFactors)
    TIME_SIGNATURES: ClassVar[tuple[str, ...]] = tuple(ts.name for ts in TimeSignatures)
    VELOCITIES: ClassVar[tuple[str, ...]] = tuple(f"Vel_{v}" for v in range(EFFECTIVE_VELOCITY_RESOLUTION))
    OFFSETS: ClassVar[tuple[str, ...]] = tuple(f"Off_Step_{o}" for o in range(-(OFFSET_STEPS+1), (OFFSET_STEPS+1) + 1))
    BPM: ClassVar[tuple[str, ...]] = tuple(f"Bpm_bin_{b}" for b in range(NUM_BPM_BINS))

    # Derived attributes (set in __post_init__)
    tokens: tuple[str, ...] = field(init=False)
    _token_to_id: Mapping[str, int] = field(init=False, repr=False)
    _id_to_token: Sequence[str] = field(init=False, repr=False)

    # Block base indices (for arithmetic)
    _base_special: int = field(init=False, repr=False)
    _base_instr: int = field(init=False, repr=False)
    _base_vel: int = field(init=False, repr=False)
    _base_off: int = field(init=False, repr=False)
    _base_grid: int = field(init=False, repr=False)
    _base_time: int = field(init=False, repr=False)
    _base_bpm: int = field(init=False, repr=False)

    # Sparse enum maps
    _inst_value_to_id: Mapping[int, int] = field(init=False, repr=False)
    _grid_enum_to_id: Mapping[GridFactors, int] = field(init=False, repr=False)
    _time_enum_to_id: Mapping[TimeSignatures, int] = field(init=False, repr=False)

    def __post_init__(self):
        # Concatenate all groups in a fixed order → stable vocab
        all_tokens = tuple(chain(
            self.SPECIAL_TOKENS,
            self.INSTRUMENTS,
            self.VELOCITIES,
            self.OFFSETS,
            self.GRID_FACTORS,
            self.TIME_SIGNATURES,
            self.BPM,
        ))

        # Safety check: no duplicates allowed
        dupes = [t for t, c in Counter(all_tokens).items() if c > 1]
        if dupes:
            raise ValueError(f"Duplicate tokens found: {dupes}")

        # Primary lookup tables
        tok2id = {tok: i for i, tok in enumerate(all_tokens)}
        object.__setattr__(self, "tokens", all_tokens)
        object.__setattr__(self, "_id_to_token", all_tokens)
        object.__setattr__(self, "_token_to_id", tok2id)

        # Compute block base indices (to support fast ID math)
        i = 0
        object.__setattr__(self, "_base_special", i)
        i += len(self.SPECIAL_TOKENS)
        object.__setattr__(self, "_base_instr", i)
        i += len(self.INSTRUMENTS)
        object.__setattr__(self, "_base_vel", i)
        i += len(self.VELOCITIES)
        object.__setattr__(self, "_base_off", i)
        i += len(self.OFFSETS)
        object.__setattr__(self, "_base_grid", i)
        i += len(self.GRID_FACTORS)
        object.__setattr__(self, "_base_time", i)
        i += len(self.TIME_SIGNATURES)
        object.__setattr__(self, "_base_bpm", i)

        # mappings for enums
        object.__setattr__(self, "_inst_value_to_id",
                           {inst.value: tok2id[inst.name] for inst in InstrumentValues})
        object.__setattr__(self, "_grid_enum_to_id",
                           {gf: tok2id[gf.name] for gf in GridFactors})
        object.__setattr__(self, "_time_enum_to_id",
                           {ts: tok2id[ts.name] for ts in TimeSignatures})

    # --- Core lookups ---
    def __getitem__(self, token: str) -> int:
        """Dictionary-style: get id for token string."""
        return self._token_to_id[token]

    def token(self, idx: int) -> str:
        """Reverse lookup: id → token string."""
        return self._id_to_token[idx]

    def __len__(self) -> int:
        return len(self.tokens)

    # --- Fast helpers ---
    @property
    def ID_BOS(self) -> int: return self._token_to_id["BOS"]

    @property
    def ID_EOS(self) -> int: return self._token_to_id["EOS"]

    @property
    def ID_BAR(self) -> int: return self._token_to_id["BAR"]

    @property
    def ID_SEP(self) -> int: return self._token_to_id["SEP"]

    def instrument_id_from_value(self, value: int) -> int:
        """Map raw instrument code to token id."""
        return self._inst_value_to_id[value]

    def vel_id(self, v: int) -> int:
        """Velocity bin → token id (0..63)."""
        return self._base_vel + v

    def off_id(self, o: int) -> int:
        """Offset step → token id. Offsets range from -OFFSET_STEPS..+OFFSET_STEPS."""
        return self._base_off + (o + OFFSET_STEPS)

    def bpm_id(self, b: int) -> int:
        """BPM bin → token id."""
        return self._base_bpm + b

    def grid_factor_id(self, gf: GridFactors) -> int:
        """Grid factor enum → token id."""
        return self._grid_enum_to_id[gf]

    def time_sig_id(self, ts: TimeSignatures) -> int:
        """Time signature enum → token id."""
        return self._time_enum_to_id[ts]
