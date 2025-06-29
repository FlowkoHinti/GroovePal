import numpy as np
import pandas as pd
import json

from torch.utils.data import Dataset
from omegaconf import DictConfig
from pathlib import Path

class DNADataset(Dataset):
    @staticmethod
    def _load_data(dna_path) -> object:
        """
        Load the DNA data.

        Parameters
        ----------
        dna_path : str
            Path to the DNA dataset file.

        Returns
        -------
        data : pd.DataFrame
            Loaded dataset.
        """
        with open(dna_path, 'rb') as f:
            data = json.load(f)

        return data

    def __init__(self, cfg: DictConfig, dataset_type: str, transform=None):
        """
        Custom Dataset for DNA sequences.

        Parameters
        ----------
        cfg : DictConfig
            Config for dataset
        dataset_type : str
            Type of dataset (train, val, test)
        transform : callable, optional
            Transformations for dataset
        """
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.transform = transform

        # Load the dataset based on the type
        self.dnas = self._load_data(cfg['dna_path'])


    def __len__(self) -> int:
        return len(self.dnas)

    def __getitem__(self, i):
        return self.dnas[i]
