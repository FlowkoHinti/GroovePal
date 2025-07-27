# Token defines
# also mask tokens/ start tokens / end tokens
from dataclasses import dataclass

from GrooveModel.Utils.BeatUnit import BEAT_UNIT_TOKEN_SIZE_ABSOLUTE, BEAT_UNIT_TOKEN_SIZE_RELATIVE
from GrooveModel.Utils.BeatsPerMinute import BPM_TOKEN_SIZE
from GrooveModel.Utils.DNAGridFactor import GRID_FACTOR_TOKEN_SIZE
from GrooveModel.Utils.TimeSignatures import TIME_SIGNATURE_TOKEN_SIZE
from GrooveModel.Utils.DNAValue import DNA_VALUE_TOKEN_SIZE
from GrooveModel.Utils.DNAVelocity import VELOCITY_TOKEN_SIZE
from GrooveModel.Utils.DNAOffset import OFFSET_TOKEN_SIZE


# vocab sizes (+1 for padding token)
INSTRUMENT_VOCAB_SIZE = DNA_VALUE_TOKEN_SIZE
VELOCITY_VOCAB_SIZE = VELOCITY_TOKEN_SIZE
OFFSET_VOCAB_SIZE = OFFSET_TOKEN_SIZE
BPM_VOCAB_SIZE = BPM_TOKEN_SIZE
TIME_SIGNATURE_VOCAB_SIZE = TIME_SIGNATURE_TOKEN_SIZE
GRID_FACTOR_VOCAB_SIZE = GRID_FACTOR_TOKEN_SIZE
BEAT_UNIT_ABSOLUTE_VOCAB_SIZE = BEAT_UNIT_TOKEN_SIZE_ABSOLUTE
BEAT_UNIT_RELATIVE_VOCAB_SIZE = BEAT_UNIT_TOKEN_SIZE_RELATIVE


@dataclass(frozen=True)
class SequentialVocab:
    # special tokens
    bos: str = 'xxbos'
    eos: str = 'xxeos'
    pad: str = 'xxpad'
    unk: str = 'xxunk'
    mask: str = 'xxmask'

    # standard vocab
    val_toks: list[str] = None