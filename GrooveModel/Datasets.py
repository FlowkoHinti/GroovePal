import json
import os
from os import PathLike
from typing import Union

from omegaconf import DictConfig
from torch.utils.data import Dataset

from GrooveModel.Tokenizer.Tokenizer import DnaTokenizer


def load_dna_json(dna_path: str):
    """Load DNA data from .json (list) or .jsonl (one JSON object per line)."""
    if dna_path.endswith(".jsonl"):
        records = []
        with open(dna_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    else:
        # .json
        with open(dna_path, "rb") as f:
            return json.load(f)  # expected to be a list


def get_all_dna_json_paths(dna_path: Union[str, PathLike]):
    """Return all DNA JSON/JSONL file paths in the directory."""
    all_files = [f for f in os.listdir(dna_path) if f.endswith(".json") or f.endswith(".jsonl")]
    if not all_files:
        raise FileNotFoundError(f'No DNA JSON/JSONL files found in {dna_path}')
    # Prefer .jsonl order first (optional), then .json
    all_files.sort(key=lambda x: (not x.endswith(".jsonl"), x))
    return [os.path.join(dna_path, f) for f in all_files]


class DNANextTokenDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str, transform=None, tokenizer: type[DnaTokenizer]=None):
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer

        tokenizer_cfg = getattr(cfg, "tokenizer", {})
        self.tokenizer_kwargs = dict(tokenizer_cfg)

        dna_dir = cfg.dna_path / split
        all_json_paths = get_all_dna_json_paths(dna_dir)

        all_dnas = []
        for json_path in all_json_paths:
            dnas = load_dna_json(json_path)
            all_dnas.extend(dnas)  # concatenate lists

        # Apply dataset subset limit if specified
        subset = getattr(cfg, "subset", None)
        if subset is not None:
            all_dnas = all_dnas[:subset]

        self.dnas = all_dnas

    def __len__(self):
        return len(self.dnas)

    def __getitem__(self, idx):
        item = self.dnas[idx]

        if self.transform:
            item = self.transform(item)

        if self.tokenizer:
            tokens = self.tokenizer.tokenize(item, **self.tokenizer_kwargs)  # (N, 7) LongTensor
        else:
            raise ValueError("Tokenizer must be provided")

        if tokens.size(0) < 2:
            raise ValueError("Token sequence too short for next-token prediction")

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        return input_tokens, target_tokens
