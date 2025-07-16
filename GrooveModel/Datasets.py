import json

from omegaconf import DictConfig
from torch.utils.data import Dataset

from GrooveModel.Tokenizers import tokens_to_tensor


def load_dna_json(dna_path: str):
    """Load DNA JSON file."""
    with open(dna_path, 'rb') as f:
        return json.load(f)


class DNANextTokenDataset(Dataset):
    def __init__(self, cfg: DictConfig, dataset_type: str, transform=None, tokenizer=None):
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.transform = transform
        self.tokenizer = tokenizer

        self.dnas = load_dna_json(cfg.dna_path)
        self.convert_to_tensor = getattr(cfg, "convert_to_tensor", False)

    def __len__(self):
        return len(self.dnas)

    def __getitem__(self, idx):
        item = self.dnas[idx]

        if self.transform:
            item = self.transform(item)

        if self.tokenizer:
            tokens = self.tokenizer.tokenize(item)
        else:
            raise ValueError("Tokenizer must be provided")

        if len(tokens) < 2:
            raise ValueError("Token sequence too short for next-token prediction")

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        if self.convert_to_tensor:
            input_tokens = tokens_to_tensor(input_tokens)
            target_tokens = tokens_to_tensor(target_tokens)

        return input_tokens, target_tokens
