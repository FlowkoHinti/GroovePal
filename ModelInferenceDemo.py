import argparse
import json
from pathlib import Path

import torch
from dacite import from_dict, Config as DaciteConfig
from omegaconf import OmegaConf, DictConfig

from Configs import BASE_PATH, MAX_SEQUENCE_LENGTH
from GrooveModel.Datasets import get_all_dna_json_paths, load_dna_json
from GrooveModel.Sampling.MultiTaskDnaSampler import MultitaskDNASampler

HEAD_ORDER = ["instrument", "velocity", "beat_unit", "offset", "grid_factor", "bpm", "time_signature"]


def main():
    ap = argparse.ArgumentParser(description="Inference demo for MultiTask DNA xLSTM (CUDA-only).")
    ap.add_argument("--config", type=str, default="experiment_initial",
                    help="Path to the experiment YAML used for training. Defaults to Experiments/experiment_initial.yml")
    args = ap.parse_args()

    # --- Load training config via OmegaConf ---
    cfg_path = BASE_PATH / "Experiments" / f"{args.config}.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    # Paths from script location
    model_dir = BASE_PATH / 'Models'
    demo_dir = BASE_PATH / 'Data' / 'demo'
    out_dir = BASE_PATH / 'Predictions'
    out_dir.mkdir(parents=True, exist_ok=True)

    dna_paths = get_all_dna_json_paths(demo_dir)
    dnas = load_dna_json(dna_paths[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sampler = MultitaskDNASampler(cfg_path, model_dir, device, use_best=True)
    result = sampler.sample(dnas[0], temperature=1.0, top_k=None, top_p=0.8, max_tokens=100)



    # --- Save combined predictions to <Data>/Predictions/predictions.json ---
    out_path = out_dir / "predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {out_path}")

# TODO UPDATE TO NEW MODEL OUTPUT HEAD AND ADD SEQUENTIAL
if __name__ == "__main__":
    main()
