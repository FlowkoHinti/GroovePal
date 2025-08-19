import json
import os
from os import PathLike
from typing import Union

from omegaconf import DictConfig
from torch.utils.data import Dataset

from GrooveModel.Tokenizer.Tokenizer import DnaTokenizer


def load_dna_json(dna_path: str):
    """Load DNA JSON file."""
    with open(dna_path, 'rb') as f:
        return json.load(f)


def get_all_dna_json_paths(dna_path: Union[str, PathLike]):
    """Return all DNA JSON file paths in the directory."""
    json_files = [f for f in os.listdir(dna_path) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f'No DNA JSON files found in {dna_path}')
    return [os.path.join(dna_path, f) for f in json_files]


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
        self.convert_to_tensor = getattr(cfg, "convert_to_tensor", False)

    def __len__(self):
        return len(self.dnas)

    def __getitem__(self, idx):
        item = self.dnas[idx]

        if self.transform:
            item = self.transform(item)

        if self.tokenizer:
            tokens = self.tokenizer.tokenize(item, **self.tokenizer_kwargs)
        else:
            raise ValueError("Tokenizer must be provided")

        if len(tokens) < 2:
            raise ValueError("Token sequence too short for next-token prediction")

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        if self.convert_to_tensor:
            input_tokens = self.tokenizer.tokens_to_tensor(input_tokens)
            target_tokens = self.tokenizer.tokens_to_tensor(target_tokens)

        return input_tokens, target_tokens
