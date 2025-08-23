# Musical Bits files already in right format -> bring bought files in order and split
# Process drumlearning foundational
import logging
import shutil
from pathlib import Path
import random

import pandas as pd
from tqdm import tqdm

from Configs import RNG_SEED
from Preprocessing.Prepper.DNAPrepper import DNAPrepper, DATA_PATH


class DrumLearningMidiPrepper(DNAPrepper):
    """
    Prepares DrumLearning dataset for fine-tuning:
      - Processes foundational/info.txt -> (.mid, .txt) pairs (excluding FULL_SONG)
      - Copies existing .mid/.txt pairs from fbo, hs, wb
      - Stages everything in raw/temp/
      - Shuffles and scatters pairs into fine_tuning/train, validation, test
    """

    def __init__(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        super().__init__()
        self.base_path = DATA_PATH / "fine_tuning"
        self.raw_path = self.base_path / "raw"
        self.temp_path = self.raw_path / "temp"
        self.foundational_meta = self._load_foundational_metadata()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def _load_foundational_metadata(self):
        meta_path = self.raw_path / "drumlearning_foundational" / "info.txt"
        meta = pd.read_csv(
            meta_path,
            header=None,
            names=["id", "variation", "set", "bpm", "type"],
            sep="\t",
        )
        meta["bpm"] = meta["bpm"].apply(lambda x: int(str(x).split(" ")[0]))
        return meta

    def prepare(self):
        logging.info("Starting DrumLearning preparation...")

        # clear/create temp staging area
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
        self.temp_path.mkdir(parents=True, exist_ok=True)

        # Step 1: prepare foundational (excluding FULL_SONG)
        foundational_pairs = self._prepare_foundational()

        # Step 2: copy fbo, hs, wb
        copied_pairs = self._copy_existing_sets(["fbo", "hs", "wb"])

        # Step 3: gather all pairs staged in temp and split
        all_pairs = foundational_pairs + copied_pairs
        logging.info(f"Total staged pairs (foundational + others): {len(all_pairs)}")

        self._split_and_scatter(all_pairs)

        # optional cleanup
        # shutil.rmtree(self.temp_path)

        logging.info("Finished DrumLearning preparation.")

    def _prepare_foundational(self):
        """
        Reads info.txt and writes .mid/.txt pairs into temp/.
        Excludes entries where type == FULL_SONG.
        Returns list of (mid_path, txt_path).
        """
        midi_src_dir = self.raw_path / "drumlearning_foundational"
        meta_filtered = self.foundational_meta[self.foundational_meta["type"] != "FULL_SONG"]

        pairs = []
        logging.info(f"Preparing foundational MIDIs from {midi_src_dir} (excluding FULL_SONG)...")

        for row in tqdm(meta_filtered.itertuples(index=False), total=len(meta_filtered)):
            mid_filename = f"{row.id}_{row.set}_{row.bpm}_{row.type}.mid"
            midi_src = midi_src_dir / mid_filename
            if not midi_src.exists():
                logging.warning(f"MIDI not found: {midi_src}")
                continue

            file_stem = f"foundational_{row.id}"
            midi_dst = self.temp_path / f"{file_stem}.mid"
            txt_dst = self.temp_path / f"{file_stem}.txt"

            # Copy midi
            shutil.copy2(midi_src, midi_dst)

            # Write txt metadata
            lines = [
                f"ID: {file_stem}",
                f"AuthorData: foundational_{row.set}",
                f"BPM: {row.bpm}",
                f"META: {row.type}",
            ]
            with open(txt_dst, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            pairs.append((midi_dst, txt_dst))

        logging.info(f"Prepared {len(pairs)} foundational pairs (excluding FULL_SONG).")
        return pairs

    def _copy_existing_sets(self, subfolders):
        """
        Copy .mid/.txt pairs from already-prepared subfolders into temp/.
        """
        pairs = []
        for sub in subfolders:
            src = self.raw_path / sub
            if not src.exists():
                logging.warning(f"Missing folder {src}, skipping.")
                continue

            for file in src.glob("*.mid"):
                txt_file = file.with_suffix(".txt")
                if not txt_file.exists():
                    logging.warning(f"Missing .txt for {file}, skipping.")
                    continue

                midi_dst = self.temp_path / file.name
                txt_dst = self.temp_path / txt_file.name
                shutil.copy2(file, midi_dst)
                shutil.copy2(txt_file, txt_dst)

                pairs.append((midi_dst, txt_dst))

            logging.info(f"Copied {len(pairs)} pairs from {sub}.")
        return pairs

    def _split_and_scatter(self, all_pairs):
        """
        Shuffle and split into train/validation/test under fine_tuning/.
        """
        random.seed(RNG_SEED)
        random.shuffle(all_pairs)

        n_total = len(all_pairs)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        n_test = n_total - n_train - n_val

        splits = {
            "train": all_pairs[:n_train],
            "validation": all_pairs[n_train : n_train + n_val],
            "test": all_pairs[n_train + n_val :],
        }

        for split, pairs in splits.items():
            split_dir = self.base_path / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)

            for mid_src, txt_src in pairs:
                shutil.copy2(mid_src, split_dir / mid_src.name)
                shutil.copy2(txt_src, split_dir / txt_src.name)

            logging.info(f"Scattered {len(pairs)} pairs into {split}/")
