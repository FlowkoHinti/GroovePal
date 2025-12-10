#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunAllEvals.py
End-to-end evaluation orchestrator.

Pipeline:
1) Run ModelInferenceDemo.py with evaluation parameters
2) Run RunMidiConverter.py to generate MIDIs
3) Run GenerateEvalData.py to move results + log metadata

Automatically resumes from the last completed experiment+parameter combination
using Evaluation/eval_progress.json checkpoint.
"""

import itertools
import json
import logging
import subprocess
import sys
from pathlib import Path

from GroovePal.Configs import BASE_PATH

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("RunAllEvals")

# -----------------------------
# Paths
# -----------------------------
MODEL_INFER = BASE_PATH / "ModelInferenceDemo.py"
PREDICTIONS_DIR = BASE_PATH / "Predictions"
MIDI_CONVERTER = PREDICTIONS_DIR / "RunMidiConverter.py"
EVAL_DIR = BASE_PATH / "Evaluation"
PROGRESS_FILE = EVAL_DIR / "eval_progress.json"
GENERATE_EVAL = EVAL_DIR / "GenerateEvalData.py"

# -----------------------------
# Experiment setup
# -----------------------------
EXPERIMENTS = [
    #"experiment_multitask_relgu_finetune",
    #"experiment_sequential_relgu_finetune",
    #"experiment_multitask_relgu_pretrain",
    #"experiment_sequential_relgu_pretrain",
    "experiment_sequential_large",
    "experiment_sequential_large_finetune",
]

# Creativeness sweep (applies to all models)
CREATIVENESS_TEMPS = [0.8, 1.0, 1.2]
CREATIVENESS_TOPPS = [0.8, 0.9, 0.95]
CREATIVENESS_MAXTOK = 200

# Sequence-length sweep (fine-tuned models only)
SEQLEN_MAXTOKS = [100, 200, 400]
SEQLEN_TEMP = 1.0
SEQLEN_TOPP = 0.9

VARIATIONS = 3  # --variations=3


# -----------------------------
# Helpers
# -----------------------------
def load_progress() -> dict | None:
    """Load last completed checkpoint."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Could not read progress file: {e}")
    return None


# -----------------------------
# Key generation and sorting
# -----------------------------
def combo_key(c: dict) -> tuple:
    """
    Generate a sortable key for a combination.
    Using a tuple ensures correct numeric ordering
    (no lexicographic string issues like '0.95' < '0.9').
    """
    return (
        c["experiment"],
        c["eval_type"],
        float(c["temperature"]),
        float(c["top_p"]),
        int(c["max_tokens"]),
    )


def build_run_plan() -> list[dict]:
    """Generate all evaluation parameter combinations."""
    plan: list[dict] = []

    # Creativeness evaluations (all models)
    for exp in EXPERIMENTS:
        for t, p in itertools.product(CREATIVENESS_TEMPS, CREATIVENESS_TOPPS):
            plan.append({
                "experiment": exp,
                "eval_type": "creativeness",
                "temperature": t,
                "top_p": p,
                "max_tokens": CREATIVENESS_MAXTOK,
            })

    # Sequence-length evaluations (only fine-tuned models)
    for exp in EXPERIMENTS:
        if "finetune" in exp:
            for n in SEQLEN_MAXTOKS:
                plan.append({
                    "experiment": exp,
                    "eval_type": "sequence_length",
                    "temperature": SEQLEN_TEMP,
                    "top_p": SEQLEN_TOPP,
                    "max_tokens": n,
                })

    return plan


def run_cmd(cmd: list[str], desc: str, cwd: Path | None = None):
    """Run a subprocess with logging and error handling."""
    log.info(f"Running step: {desc}")
    log.debug(f"Command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Step '{desc}' failed with exit code {proc.returncode}")


# -----------------------------
# Main
# -----------------------------
def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    run_plan = build_run_plan()
    run_plan.sort(key=combo_key)
    progress = load_progress()
    resume_after = None
    if progress and progress.get("last_completed"):
        resume_after = progress.get("last_completed") if progress else None
        log.info(f"Resuming after: {resume_after}")

    for combo in run_plan:
        key = combo_key(combo)
        if resume_after and key <= combo_key(resume_after):
            continue  # skip already completed

        exp = combo["experiment"]
        etype = combo["eval_type"]
        t = str(combo["temperature"])
        p = str(combo["top_p"])
        n = str(combo["max_tokens"])

        log.info("=" * 80)
        log.info(f"Starting evaluation: {key}")
        log.info("=" * 80)

        # Step 1: Model inference (generates JSONs)
        run_cmd(
            [
                sys.executable, str(MODEL_INFER),
                "--config", exp,
                "--variations", str(VARIATIONS),
                "--max_tokens", n,
                "--temperature", t,
                "--top_p", p,
                "--eval", "True",
            ],
            desc="Model Inference",
            cwd=BASE_PATH,
        )

        # Step 2: Convert JSONs to MIDIs
        run_cmd(
            [sys.executable, str(MIDI_CONVERTER)],
            desc="MIDI Conversion",
            cwd=PREDICTIONS_DIR,
        )

        # Step 3: Archive results and update metadata
        run_cmd(
            [
                sys.executable, str(GENERATE_EVAL),
                "--experiment", exp,
                "--eval_type", etype,
                "--temperature", t,
                "--top_p", p,
                "--max_tokens", n,
                "--resume",
            ],
            desc="Generate Evaluation Data",
            cwd=EVAL_DIR,
        )

        log.info(f"Completed: {key}")

    log.info("All evaluations finished successfully.")


if __name__ == "__main__":
    main()
