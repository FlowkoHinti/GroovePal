import io
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from datasets import load_dataset
from pretty_midi import pretty_midi
from tqdm import tqdm

from Configs import RNG_SEED

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
BASE_PATH = Path.cwd()
DATA_PATH = BASE_PATH / 'Data'

# Max number of MIDI/TXT PAIRS per subfolder
MAX_PAIRS_PER_CHUNK = 25_000


class DNAPrepper(ABC):
    @abstractmethod
    def prepare(self):
        """Main preparation pipeline."""
        pass


class GigaMidiPrepper(DNAPrepper):
    def __init__(self, train_size=100000, val_size=10000, test_size=10000):
        with open(BASE_PATH.parent / 'api_keys', 'r') as file:
            file_content = file.read().strip()
            os.environ["api_key"] = file_content

        self.dataset = load_dataset("Metacreation/GigaMIDI", token=os.environ["api_key"])
        self.dataset = self.dataset.filter(lambda sample: sample["instrument_category"] == 'drums-only')
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def prepare(self):
        train_subset = self.dataset["train"].shuffle(seed=RNG_SEED).select(range(self.train_size))
        validation_subset = self.dataset["validation"].shuffle(seed=RNG_SEED).select(range(self.val_size))
        test_subset = self.dataset["test"].shuffle(seed=RNG_SEED).select(range(self.test_size))

        self.prepare_midis(train_subset, 'train')
        self.prepare_midis(validation_subset, 'validation')
        self.prepare_midis(test_subset, 'test')

    @staticmethod
    def prepare_midis(subset, destination: str):
        """
        Writes valid MIDI samples as (.mid, .txt) PAIRS into chunked subfolders:
        Data/pre_training/<destination>/chunk_0000, chunk_0001, ...
        Each chunk contains at most MAX_PAIRS_PER_CHUNK pairs.
        """
        destination_path = DATA_PATH / 'pre_training' / destination
        os.makedirs(destination_path, exist_ok=True)

        logging.info(f"Preparing {len(subset)} MIDI files for '{destination}' set...")

        valid_count = 0
        for i, sample in enumerate(tqdm(subset, desc=f"Processing {destination}")):
            midi_bytes = sample['music']
            title = sample.get('title_scraped')
            artist = sample.get('artist_scraped')

            try:
                midi_obj = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
                note_count = sum(len(instr.notes) for instr in midi_obj.instruments)
                if note_count == 0:
                    logging.debug(f"Skipped MIDI with no notes (index {i}).")
                    continue
            except Exception as e:
                logging.warning(f"Malformed MIDI at index {i}: {e}")
                continue

            # Decide chunk folder based on sequential VALID pair index
            chunk_index = valid_count // MAX_PAIRS_PER_CHUNK
            chunk_path = destination_path / f'chunk_{chunk_index:04d}'
            chunk_path.mkdir(parents=True, exist_ok=True)

            # Keep global, sequential stems so IDs are unique across chunks
            file_stem = f'giga_{valid_count}'
            valid_count += 1

            midi_path = chunk_path / f'{file_stem}.mid'
            txt_path = chunk_path / f'{file_stem}.txt'

            # Write MIDI
            with open(midi_path, 'wb') as f:
                f.write(midi_bytes)

            # Write metadata
            lines = [f'ID: {file_stem}']
            if title and artist:
                lines.append(f'AuthorData: {title} {artist}')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            logging.debug(f"Wrote {midi_path} and {txt_path}")

        logging.info(
            f"Finished processing. {valid_count} valid MIDI files saved to '{destination}' in chunked subfolders."
        )