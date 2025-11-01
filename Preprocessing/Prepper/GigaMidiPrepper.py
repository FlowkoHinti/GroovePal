import logging
import os
import io

from datasets import load_dataset
import pretty_midi
from tqdm import tqdm

from Configs import RNG_SEED
from Preprocessing.Prepper.DNAPrepper import DNAPrepper, BASE_PATH, DATA_PATH, MAX_PAIRS_PER_CHUNK


INSTRUMENT_CATEGORY_KEY = "instrument_category"
EXPRESSIVE_THRESHOLD = 12  # NOMML threshold via median_metric_depth

class GigaMidiPrepper(DNAPrepper):
    def __init__(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, expressive_only=True):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Train/val/test ratios must sum to 1.0"

        # HF auth
        with open(BASE_PATH.parent / 'api_keys', 'r') as f:
            os.environ["api_key"] = f.read().strip()

        # Load all splits together
        raw = load_dataset(
            "Metacreation/GigaMIDI",
            token=os.environ["api_key"],
            split="train+validation+test"
        )

        # 1) Keep only drums-only files
        drums_only = raw.filter(lambda s: s.get(INSTRUMENT_CATEGORY_KEY) == "drums-only")

        # 2) Among those, "expressive" means any NOMML value in median_metric_depth >= 12
        def is_expressive(sample):
            md = sample.get("median_metric_depth")
            if not isinstance(md, list) or len(md) == 0:
                return False
            try:
                return max(md) >= EXPRESSIVE_THRESHOLD
            except Exception:
                return False

        expressive_drums = drums_only.filter(is_expressive)

        # Print + log counts (all drums vs expressive drums)
        total_drums = len(drums_only)
        total_expressive_drums = len(expressive_drums)
        print(f"Total drums-only files: {total_drums}")
        print(f"Total expressive drums-only (median_metric_depth >= {EXPRESSIVE_THRESHOLD} on any track): {total_expressive_drums}")
        logging.info(f"Total drums-only files: {total_drums}")
        logging.info(f"Total expressive drums-only (>= {EXPRESSIVE_THRESHOLD}): {total_expressive_drums}")

        # Choose working set for writing
        self.dataset = expressive_drums if expressive_only else drums_only

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def prepare(self):
        shuffled = self.dataset.shuffle(seed=RNG_SEED)
        total = len(shuffled)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        train_subset = shuffled.select(range(0, train_end))
        val_subset   = shuffled.select(range(train_end, val_end))
        test_subset  = shuffled.select(range(val_end, total))

        logging.info(f"Split sizes — train: {len(train_subset)}, val: {len(val_subset)}, test: {len(test_subset)}")

        self.prepare_midis(train_subset, 'train')
        self.prepare_midis(val_subset, 'validation')
        self.prepare_midis(test_subset, 'test')

    @staticmethod
    def prepare_midis(subset, destination: str):
        """
        Writes valid (.mid, .txt) pairs into chunked subfolders:
        Data/pre_training/<destination>/chunk_0000, chunk_0001, ...
        Each chunk contains at most MAX_PAIRS_PER_CHUNK pairs.
        """
        destination_path = DATA_PATH / 'pre_training' / destination
        os.makedirs(destination_path, exist_ok=True)

        logging.info(f"Preparing {len(subset)} MIDI files for '{destination}' set...")

        valid_count = 0
        for i, sample in enumerate(tqdm(subset, desc=f"Processing {destination}")):
            midi_bytes = sample['music']
            title = sample.get('title_scraped') or sample.get('title')
            artist = sample.get('artist_scraped') or sample.get('artist')

            try:
                midi_obj = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
                note_count = sum(len(instr.notes) for instr in midi_obj.instruments)
                if note_count == 0:
                    logging.debug(f"Skipped MIDI with no notes (index {i}).")
                    continue
            except Exception as e:
                logging.warning(f"Malformed MIDI at index {i}: {e}")
                continue

            chunk_index = valid_count // MAX_PAIRS_PER_CHUNK
            chunk_path = destination_path / f'chunk_{chunk_index:04d}'
            chunk_path.mkdir(parents=True, exist_ok=True)

            file_stem = f'giga_{valid_count}'
            valid_count += 1

            midi_path = chunk_path / f'{file_stem}.mid'
            txt_path  = chunk_path / f'{file_stem}.txt'

            with open(midi_path, 'wb') as f:
                f.write(midi_bytes)

            lines = [f'ID: {file_stem}']
            if title and artist:
                lines.append(f'AuthorData: {title} {artist}')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

        logging.info(f"Finished processing. {valid_count} valid MIDI files saved to '{destination}'.")
