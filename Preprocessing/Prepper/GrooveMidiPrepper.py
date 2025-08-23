import logging
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from pretty_midi import pretty_midi

from Preprocessing.Prepper.DNAPrepper import (
    DNAPrepper,
    DATA_PATH,
    MAX_PAIRS_PER_CHUNK,
)

class GrooveMidiPrepper(DNAPrepper):
    """
    Prepares the Groove (E-GMD) dataset into chunked (<MAX_PAIRS_PER_CHUNK>) pairs of
    (.mid, .txt) under:
        Data/intermediate/{train|validation|test}/chunk_0000, chunk_0001, ...

    For each valid MIDI file, we write:
      - <chunk_path>/groove_{global_index}.mid
      - <chunk_path>/groove_{global_index}.txt
        with lines:
            ID: groove_{global_index}
            AuthorData: drummer_<drummer_id>
            BPM: <bpm>
            Numerator: <num>
            Denominator: <den>
    """

    def __init__(self, groove_root: Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.groove_path = (
            groove_root
            if groove_root is not None
            else DATA_PATH / "intermediate" / "raw" / "e-gmd-v1.0.0"
        )
        self.metadata: pd.DataFrame = self._load_groove_meta()

    def _load_groove_meta(self) -> pd.DataFrame:
        meta_csv = self.groove_path / "e-gmd-v1.0.0.csv"
        if not meta_csv.exists():
            raise FileNotFoundError(f"Groove metadata CSV not found at: {meta_csv}")
        return pd.read_csv(meta_csv)

    def prepare(self):
        for split in ("train", "validation", "test"):
            split_df = self.metadata[self.metadata["split"] == split]
            self._prepare_split(split_df, split)

    @staticmethod
    def _parse_time_signature(ts: str):
        try:
            num_s, den_s = ts.split("-", maxsplit=1)
            return int(num_s), int(den_s)
        except Exception:
            return None, None

    @staticmethod
    def _safe_int(x):
        try:
            return int(x)
        except Exception:
            return None

    def _prepare_split(self, split_df: pd.DataFrame, destination: str):
        """
        Writes valid MIDI samples as (.mid, .txt) pairs into chunked subfolders:
        Data/intermediate/<destination>/chunk_0000, chunk_0001, ...
        """
        destination_path = DATA_PATH / "intermediate" / destination
        os.makedirs(destination_path, exist_ok=True)

        logging.info(f"Preparing {len(split_df)} Groove MIDIs for '{destination}' set...")

        valid_count = 0
        for i, row in enumerate(
            tqdm(split_df.itertuples(index=False), total=len(split_df), desc=f"Processing {destination}")
        ):
            midi_rel = getattr(row, "midi_filename", None)
            midi_path_src = self.groove_path / midi_rel
            if not midi_rel or not midi_path_src.exists():
                continue

            # Validate: parse + has notes
            try:
                midi_obj = pretty_midi.PrettyMIDI(str(midi_path_src))
                note_count = sum(len(instr.notes) for instr in midi_obj.instruments)
                if note_count == 0:
                    continue
            except Exception as e:
                logging.warning(f"Malformed MIDI at row {i} ({midi_path_src}): {e}")
                continue

            # Decide chunk folder based on sequential valid index
            chunk_index = valid_count // MAX_PAIRS_PER_CHUNK
            chunk_path = destination_path / f"chunk_{chunk_index:04d}"
            chunk_path.mkdir(parents=True, exist_ok=True)

            file_stem = f"groove_{valid_count}"
            valid_count += 1

            midi_path_dst = chunk_path / f"{file_stem}.mid"
            txt_path = chunk_path / f"{file_stem}.txt"

            # Copy MIDI
            with open(midi_path_src, "rb") as fr, open(midi_path_dst, "wb") as fw:
                fw.write(fr.read())

            # Metadata lines
            drummer = getattr(row, "drummer", None)
            bpm = self._safe_int(getattr(row, "bpm", None))
            num, den = self._parse_time_signature(getattr(row, "time_signature", None))

            lines = [f"ID: {file_stem}"]
            if drummer:
                lines.append(f"AuthorData: drummer_{drummer}")
            if bpm is not None:
                lines.append(f"BPM: {bpm}")
            if num is not None:
                lines.append(f"Numerator: {num}")
            if den is not None:
                lines.append(f"Denominator: {den}")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        logging.info(
            f"Finished processing. {valid_count} valid MIDI files saved to '{destination}' in chunked subfolders."
        )
