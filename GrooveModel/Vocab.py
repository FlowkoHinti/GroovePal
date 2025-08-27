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


@dataclass(slots=True, frozen=True)
class SequentialDnaVocab:
    """
    A frozen, bidirectional vocabulary for sequential DNA-style music tokenization.

    - Concatenates several token groups in a fixed order -> stable IDs.
    - Builds fast lookups (token -> id, id -> token).
    - Exposes base indices for contiguous blocks to allow O(1) ID arithmetic.
    """

    # --- Static token groups (immutable by design) ---
    SPECIAL_TOKENS: ClassVar[tuple[str, ...]] = ("PAD", "BOS", "EOS", "MASK", "SEP", "BAR", "CLS")
    INSTRUMENTS: ClassVar[tuple[str, ...]] = tuple(i.name for i in InstrumentValues)
    GRID_FACTORS: ClassVar[tuple[str, ...]] = tuple(gf.name for gf in GridFactors)
    TIME_SIGNATURES: ClassVar[tuple[str, ...]] = tuple(ts.name for ts in TimeSignatures)
    VELOCITIES: ClassVar[tuple[str, ...]] = tuple(f"Vel_{v}" for v in range(EFFECTIVE_VELOCITY_RESOLUTION))
    OFFSETS: ClassVar[tuple[str, ...]] = tuple(
        f"Off_Step_{o}" for o in range(-(OFFSET_STEPS + 1), (OFFSET_STEPS + 1) + 1)
    )
    BPM: ClassVar[tuple[str, ...]] = tuple(f"Bpm_bin_{b}" for b in range(NUM_BPM_BINS))

    # --- Derived at init ---
    tokens: tuple[str, ...] = field(init=False)
    _token_to_id: Mapping[str, int] = field(init=False, repr=False)
    _id_to_token: Sequence[str] = field(init=False, repr=False)

    # Block base indices (the starting ID for each group)
    _base_special: int = field(init=False, repr=False)
    _base_instr: int = field(init=False, repr=False)
    _base_vel: int = field(init=False, repr=False)
    _base_off: int = field(init=False, repr=False)
    _base_grid: int = field(init=False, repr=False)
    _base_time: int = field(init=False, repr=False)
    _base_bpm: int = field(init=False, repr=False)

    # Enum -> ID maps (sparse lookups)
    _inst_value_to_id: Mapping[InstrumentValues, int] = field(init=False, repr=False)
    _grid_enum_to_id: Mapping[GridFactors, int] = field(init=False, repr=False)
    _time_enum_to_id: Mapping[TimeSignatures, int] = field(init=False, repr=False)

    # ------------------------------ init ------------------------------

    def __post_init__(self) -> None:
        # 1) Build the full ordered token list
        all_tokens = self._build_all_tokens()

        # 2) Safety guard against accidental duplicates
        self._ensure_no_duplicates(all_tokens)

        # 3) Primary lookup tables
        tok2id = {tok: i for i, tok in enumerate(all_tokens)}
        object.__setattr__(self, "tokens", all_tokens)
        object.__setattr__(self, "_id_to_token", all_tokens)
        object.__setattr__(self, "_token_to_id", tok2id)

        # 4) Compute base indices for each block (for O(1) arithmetic)
        bases = self._compute_block_bases()
        for name, value in bases.items():
            object.__setattr__(self, name, value)

        # 5) Enum -> ID maps
        object.__setattr__(self, "_inst_value_to_id", {inst: tok2id[inst.name] for inst in InstrumentValues})
        object.__setattr__(self, "_grid_enum_to_id", {gf: tok2id[gf.name] for gf in GridFactors})
        object.__setattr__(self, "_time_enum_to_id", {ts: tok2id[ts.name] for ts in TimeSignatures})

    # --------------------------- builders -----------------------------

    def _build_all_tokens(self) -> tuple[str, ...]:
        """Concatenate groups in a fixed order to yield a stable vocabulary."""
        return tuple(
            chain(
                self.SPECIAL_TOKENS,
                self.INSTRUMENTS,
                self.VELOCITIES,
                self.OFFSETS,
                self.GRID_FACTORS,
                self.TIME_SIGNATURES,
                self.BPM,
            )
        )

    @staticmethod
    def _ensure_no_duplicates(tokens: Sequence[str]) -> None:
        dupes = [t for t, c in Counter(tokens).items() if c > 1]
        if dupes:
            raise ValueError(f"Duplicate tokens found: {dupes}")

    def _compute_block_bases(self) -> dict[str, int]:
        """
        Compute and return starting indices for each token block.
        The order must match _build_all_tokens().
        """
        i = 0
        bases: dict[str, int] = {}

        bases["_base_special"] = i
        i += len(self.SPECIAL_TOKENS)

        bases["_base_instr"] = i
        i += len(self.INSTRUMENTS)

        bases["_base_vel"] = i
        i += len(self.VELOCITIES)

        bases["_base_off"] = i
        i += len(self.OFFSETS)

        bases["_base_grid"] = i
        i += len(self.GRID_FACTORS)

        bases["_base_time"] = i
        i += len(self.TIME_SIGNATURES)

        bases["_base_bpm"] = i
        # i += len(self.BPM)  # not needed beyond this point

        return bases

    # --------------------------- core api -----------------------------

    def __getitem__(self, token: str) -> int:
        """Dictionary-style: token string -> ID (e.g., vocab['EOS'])."""
        return self._token_to_id[token]

    def token(self, idx: int) -> str:
        """Reverse lookup: ID -> token string."""
        return self._id_to_token[idx]

    def __len__(self) -> int:
        return len(self.tokens)

    # ------------------------- convenience ids ------------------------

    @property
    def ID_BOS(self) -> int:
        return self._token_to_id["BOS"]

    @property
    def ID_EOS(self) -> int:
        return self._token_to_id["EOS"]

    @property
    def ID_BAR(self) -> int:
        return self._token_to_id["BAR"]

    @property
    def ID_SEP(self) -> int:
        return self._token_to_id["SEP"]

    # --------------------------- fast helpers -------------------------

    def instrument_id_from_value(self, value: InstrumentValues) -> int:
        """Instrument enum -> token ID."""
        return self._inst_value_to_id[value]

    def vel_id(self, v: int) -> int:
        """Velocity bin (0..EFFECTIVE_VELOCITY_RESOLUTION-1) -> token ID."""
        return self._base_vel + v

    def off_id(self, o: int) -> int:
        """Offset step (-OFFSET_STEPS..+OFFSET_STEPS) -> token ID."""
        return self._base_off + (o + OFFSET_STEPS)

    def bpm_id(self, b: int) -> int:
        """BPM bin -> token ID."""
        return self._base_bpm + b

    def grid_factor_id(self, gf: GridFactors) -> int:
        """Grid factor enum -> token ID."""
        return self._grid_enum_to_id[gf]

    def time_sig_id(self, ts: TimeSignatures) -> int:
        """Time signature enum -> token ID."""
        return self._time_enum_to_id[ts]