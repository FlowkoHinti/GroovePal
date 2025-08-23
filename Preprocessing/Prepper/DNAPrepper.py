import logging
from abc import ABC, abstractmethod

from Configs import BASE_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
DATA_PATH = BASE_PATH / 'Data'

# Max number of MIDI/TXT PAIRS per subfolder
MAX_PAIRS_PER_CHUNK = 2_000


class DNAPrepper(ABC):
    @abstractmethod
    def prepare(self):
        """Main preparation pipeline."""
        pass
