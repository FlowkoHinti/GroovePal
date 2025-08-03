import json
import logging
from pathlib import Path
from tqdm import tqdm

# Logging setup (already present in your code)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
BASE_PATH = Path.cwd()
DATA_PATH = BASE_PATH / 'Data'


def clean_dna_file(json_path: Path):
    logging.info(f"Processing: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
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

    original_count = len(data)
    cleaned_data = []
    removed_count = 0

    for dna in data:
        dna_units = dna.get('DNAUnits', [])
        if not isinstance(dna_units, list) or len(dna_units) < 10:
            removed_count += 1
            continue

        dna['MusicEvents'] = None
        cleaned_data.append(dna)

    if not cleaned_data:
        logging.info(f"All entries removed from {json_path}. Skipping write.")
        return

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Cleaned {json_path}: {original_count - removed_count} kept, {removed_count} removed.")
    except Exception as e:
        logging.error(f"Failed to write cleaned data to {json_path}: {e}")


def process_all_jsons(data_path: Path):
    json_files = list(data_path.rglob('*.json'))
    logging.info(f"Found {len(json_files)} JSON files in {data_path}.")

    for json_file in tqdm(json_files, desc="Processing JSONs"):
        clean_dna_file(json_file)


# Run the script
if __name__ == "__main__":
    process_all_jsons(DATA_PATH)