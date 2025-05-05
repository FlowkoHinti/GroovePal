# Token defines
# also mask tokens/ start tokens / end tokens
from pydantic.v1.dataclasses import dataclass


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