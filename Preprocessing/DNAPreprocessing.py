import json
import logging
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from GrooveModel.Utils.BeatsPerMinute import MIN_BPM, MAX_BPM

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
BASE_PATH = Path.cwd()
DATA_PATH = BASE_PATH / "Data"

# Keep only these top-level song keys
KEEP_KEYS = [
    "DNA_ID", "StyleTags", "DNAUnits", "Bpm", "GridFactor",
    "Numerator", "Denominator", "TicksPerQuarterNote",
    "NumberOfBars", "TicksPerGridUnit", "FillStartTicks"
]

# Keep only these keys inside each DNA unit
DNAUNIT_KEEP_KEYS = [
    "Value", "ExcludeValue", "Wildcard", "AvgOffsetTicks", "AvgVelocity",
    "OffsetTicksPerValuePart", "VelocityPerValuePart", "IsEmpty"
]

# Allowed time signatures (Numerator, Denominator)
VALID_TIME_SIGNATURES = {
    (4, 4), (3, 4), (6, 8), (2, 4), (2, 2),
    (5, 4), (7, 8), (9, 8), (12, 8), (3, 8),
    (6, 4), (3, 2)
}


def count_leading_empty_units(dna_units):
    """Count ONLY the consecutive leading units with IsEmpty == True."""
    count = 0
    for u in dna_units:
        if not isinstance(u, dict) or not u.get("IsEmpty", False):
            break
        count += 1
    return count


def clean_dna_file(json_path: Path, index_in_folder: int):
    logging.info(f"Processing: {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in {json_path}: {e}")
        return
    except Exception as e:
        logging.error(f"Unexpected error reading {json_path}: {e}")
        return

    if not isinstance(data, list):
        logging.warning(f"Skipping {json_path}: Expected a list at root.")
        return

    cleaned_data = []
    removed_count = 0

    for dna in data:
        dna_units = dna.get("DNAUnits", [])
        if not isinstance(dna_units, list):
            removed_count += 1
            continue

        # Effective length = total units minus the number of LEADING empty units
        leading_empty = count_leading_empty_units(dna_units)
        effective_length = len(dna_units) - leading_empty
        if effective_length < 10:
            removed_count += 1
            continue

        bpm = dna.get("Bpm", 0)
        if not isinstance(bpm, int) or bpm < MIN_BPM or bpm >= MAX_BPM:
            removed_count += 1
            continue

        num, den = dna.get("Numerator", 0), dna.get("Denominator", 0)
        if (num, den) not in VALID_TIME_SIGNATURES:
            removed_count += 1
            continue

        # Keep only allowed top-level keys
        new_dna = {k: dna[k] for k in KEEP_KEYS if k in dna}

        # Keep ALL DNA units but drop the two disallowed keys
        new_dna["DNAUnits"] = [
            {k: unit[k] for k in DNAUNIT_KEEP_KEYS if k in unit}
            if isinstance(unit, dict) else unit
            for unit in dna_units
        ]

        cleaned_data.append(new_dna)

    if not cleaned_data:
        logging.info(f"All entries removed from {json_path}. Skipping write/rename.")
        return

    # Rename to <foldername>_chunk_x.jsonl
    foldername = json_path.parent.name
    new_filename = f"{foldername}_chunk_{index_in_folder}.jsonl"
    new_path = json_path.with_name(new_filename)

    try:
        with open(new_path, "w", encoding="utf-8") as f:
            for item in cleaned_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Delete the original file if different name
        if new_path.resolve() != json_path.resolve():
            try:
                json_path.unlink()
            except Exception as e:
                logging.warning(f"Could not delete original file {json_path}: {e}")

        logging.info(
            f"Cleaned {json_path} -> {new_path}: {len(cleaned_data)} kept, {removed_count} removed."
        )
    except Exception as e:
        logging.error(f"Failed to write cleaned data to {new_path}: {e}")


def process_all_jsons(data_path: Path):
    # Deterministic order
    json_files = sorted(data_path.rglob("*.json"))
    logging.info(f"Found {len(json_files)} JSON files in {data_path}.")

    # Per-folder counters so x is the index among siblings
    per_folder_index = defaultdict(int)

    for json_file in tqdm(json_files, desc="Processing JSONs"):
        parent = json_file.parent
        per_folder_index[parent] += 1
        clean_dna_file(json_file, per_folder_index[parent])


# Run the script
if __name__ == "__main__":
    process_all_jsons(DATA_PATH)