#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GenerateEvalData.py
Organizes generated DNA JSON + MIDI files from Predictions/ into Evaluation/EvalData/,
and logs metadata in eval_metadata_old.csv. Supports resume and restart checkpointing.
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from GroovePal.Configs import BASE_PATH

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
PREDICTIONS_DIR = BASE_PATH / "Predictions"
EVAL_DIR = BASE_PATH / "Evaluation"
EVALDATA_DIR = EVAL_DIR / "EvalData"
PROGRESS_FILE = EVAL_DIR / "eval_progress.json"
METADATA_CSV = EVAL_DIR / "eval_metadata_old.csv"

ORIGINAL_INPUTS_DIR = EVALDATA_DIR / "OriginalInputs"
ORIGINAL_INPUTS_DNA = ORIGINAL_INPUTS_DIR / "dna"
ORIGINAL_INPUTS_MIDI = ORIGINAL_INPUTS_DIR / "midi"


# -----------------------------
# Helpers
# -----------------------------
def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def is_variation(filename: str) -> bool:
    return "_variation_" in filename


def parse_variation(filename: str) -> Optional[int]:
    try:
        return int(filename.split("_variation_")[-1].split(".")[0])
    except Exception:
        return None


def model_info_from_experiment(experiment: str) -> Dict[str, str]:
    model = experiment.replace("experiment_", "")
    model_type = "multitask" if "multitask" in model else "sequential"
    pretrain_type = "pretrain" if "pretrain" in model or not "finetune" in model else "finetune"
    return {"model": model, "model_type": model_type, "pretrain_type": pretrain_type}


def atomic_write_json(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_json_safely(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path.name}: {e}")
        return None


def count_dna_units(dna_json: dict) -> int:
    try:
        return len(dna_json[0].get("DNAUnits", []))
    except Exception:
        return 0


def load_progress() -> Optional[dict]:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def update_progress(entry: dict):
    atomic_write_json(PROGRESS_FILE, {"last_completed": entry, "status": "complete"})


# -----------------------------
# Main processing
# -----------------------------
def process_predictions(args):
    safe_mkdir(EVALDATA_DIR)
    safe_mkdir(ORIGINAL_INPUTS_DNA)
    safe_mkdir(ORIGINAL_INPUTS_MIDI)

    info = model_info_from_experiment(args.experiment)
    model, model_type, pretrain_type = info["model"], info["model_type"], info["pretrain_type"]

    # Skip pretrain models for sequence_length eval
    if args.eval_type == "sequence_length" and pretrain_type == "pretrain":
        logger.info(f"Skipping pretrain model '{model}' for sequence_length evals.")
        return

    # Define experiment parameter folder
    folder_name = f"{model}"
    if args.eval_type == "creativeness":
        folder_name += f"_temp{args.temperature}_top_p{args.top_p}"
    elif args.eval_type == "sequence_length":
        folder_name += f"_maxtok{args.max_tokens}"

    dest_root = EVALDATA_DIR / folder_name
    dna_dest = dest_root / "dna"
    midi_dest = dest_root / "midi"
    safe_mkdir(dna_dest)
    safe_mkdir(midi_dest)

    json_files = list(PREDICTIONS_DIR.glob("*.json"))
    if not json_files:
        logger.info(f"No JSON files found in {PREDICTIONS_DIR}")
        return

    metadata_rows = []

    for json_path in json_files:
        midi_path = json_path.with_suffix(".mid")
        filename = json_path.name

        if not is_variation(filename):
            dest_json = ORIGINAL_INPUTS_DNA / filename
            dest_midi = ORIGINAL_INPUTS_MIDI / midi_path.name

            if dest_json.exists():
                continue
            shutil.move(json_path, dest_json)
            if midi_path.exists():
                shutil.move(midi_path, dest_midi)
            continue  # originals not added as rows

        # Variations
        variation_idx = parse_variation(filename)
        data = load_json_safely(json_path)
        unit_count = count_dna_units(data) if data else 0

        dest_json = dna_dest / filename
        dest_midi = midi_dest / midi_path.name

        shutil.move(json_path, dest_json)
        if midi_path.exists():
            shutil.move(midi_path, dest_midi)

        input_json_name = filename.split("_variation_")[0] + ".json"
        input_midi_name = filename.split("_variation_")[0] + ".mid"
        input_dna_rel = f"EvalData/OriginalInputs/dna/{input_json_name}"
        input_midi_rel = f"EvalData/OriginalInputs/midi/{input_midi_name}"

        row = {
            "input_dna": input_dna_rel,
            "input_midi": input_midi_rel,
            "variation": variation_idx,
            "model": model,
            "model_type": model_type,
            "pretrain_type": pretrain_type,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "eval_type": args.eval_type,
            "unit_count": unit_count,
            "dna": f"EvalData/{folder_name}/dna/{filename}",
            "midi": f"EvalData/{folder_name}/midi/{midi_path.name}",
        }
        metadata_rows.append(row)

    if not metadata_rows:
        logger.info("No variations processed.")
        return

    df_new = pd.DataFrame(metadata_rows)
    if METADATA_CSV.exists():
        df_old = pd.read_csv(METADATA_CSV)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(METADATA_CSV, index=False)
    logger.info(f"Metadata updated: {METADATA_CSV} ({len(df_new)} new rows)")

    progress_entry = {
        "experiment": args.experiment,
        "eval_type": args.eval_type,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    update_progress(progress_entry)
    logger.info(f"Progress checkpoint updated: {PROGRESS_FILE}")


# -----------------------------
# Main entrypoint
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Organize model predictions into evaluation data structure.")
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--eval_type", type=str, choices=["creativeness", "sequence_length"], required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    progress = None
    if args.restart and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        logger.info("Previous progress checkpoint removed (restart mode).")

    if args.resume and not args.restart:
        progress = load_progress()

    if progress:
        last = progress.get("last_completed", {})
        if last == {
            "experiment": args.experiment,
            "eval_type": args.eval_type,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }:
            logger.info("This combination already completed. Skipping.")
            return

    process_predictions(args)
    logger.info("Processing complete.")


if __name__ == "__main__":
    main()