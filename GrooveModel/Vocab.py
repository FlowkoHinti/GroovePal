# Token defines
# also mask tokens/ start tokens / end tokens
from dataclasses import dataclass, field
from itertools import chain
from typing import ClassVar, Mapping, Sequence, Counter

from GrooveModel.Utils.BeatUnit import BEAT_UNIT_TOKEN_SIZE_ABSOLUTE, BEAT_UNIT_TOKEN_SIZE_RELATIVE
from GrooveModel.Utils.BeatsPerMinute import BPM_TOKEN_SIZE, NUM_BPM_BINS
from GrooveModel.Utils.DNAGridFactor import GRID_FACTOR_TOKEN_SIZE, GridFactors
from GrooveModel.Utils.DNAOffset import OFFSET_TOKEN_SIZE, OFFSET_TICKS_RESOLUTION
from GrooveModel.Utils.DNAValue import DNA_VALUE_TOKEN_SIZE, InstrumentValues
from GrooveModel.Utils.DNAVelocity import VELOCITY_TOKEN_SIZE, VELOCITY_RESOLUTION
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
    # Immutable token groups
    SPECIAL_TOKENS: ClassVar[tuple[str, ...]] = ("PAD", "BOS", "EOS", "MASK", "SEP", "CLS")
    INSTRUMENTS: ClassVar[tuple[str, ...]] = tuple(i.name for i in InstrumentValues)
    GRID_FACTORS: ClassVar[tuple[str, ...]] = tuple(g.name for g in GridFactors)
    TIME_SIGNATURES: ClassVar[tuple[str, ...]] = tuple(ts.name for ts in TimeSignatures)
    VELOCITIES: ClassVar[tuple[str, ...]] = tuple(f"Vel_{v}" for v in range(VELOCITY_RESOLUTION))
    OFFSETS: ClassVar[tuple[str, ...]] = tuple(f"Off_{o}" for o in range(OFFSET_TICKS_RESOLUTION))
    BPM: ClassVar[tuple[str, ...]] = tuple(f"Bpm_bin_{b}" for b in range(NUM_BPM_BINS))

    # Derived at construction
    tokens: tuple[str, ...] = field(init=False)
    _token_to_id: Mapping[str, int] = field(init=False, repr=False)
    _id_to_token: Sequence[str] = field(init=False, repr=False)

    def __post_init__(self):
        all_tokens = tuple(chain(
            self.SPECIAL_TOKENS,
            self.INSTRUMENTS,
            self.VELOCITIES,
            self.OFFSETS,
            self.GRID_FACTORS,
            self.TIME_SIGNATURES,
            self.BPM,
        ))

        # Optional safety check for duplicates
        dupes = [t for t, c in Counter(all_tokens).items() if c > 1]
        if dupes:
            raise ValueError(f"Duplicate tokens found: {dupes}")

        object.__setattr__(self, "tokens", all_tokens)
        object.__setattr__(self, "_id_to_token", all_tokens)
        object.__setattr__(self, "_token_to_id", {tok: i for i, tok in enumerate(all_tokens)})

    # Dict-style token -> id
    def __getitem__(self, token: str) -> int:
        return self._token_to_id[token]

    # Reverse lookup
    def token(self, idx: int) -> str:
        return self._id_to_token[idx]

    # Vocab size
    def __len__(self) -> int:
        return len(self.tokens)
