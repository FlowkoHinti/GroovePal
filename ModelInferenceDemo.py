import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
from pathlib import Path

import torch
from dacite import from_dict, Config as DaciteConfig
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from Configs import BASE_PATH, MAX_SEQUENCE_LENGTH
from GrooveModel.Datasets import get_all_dna_json_paths, load_dna_json
from GrooveModel.Sampling.MultiTaskDnaSampler import MultitaskDNASampler
from GrooveModel.Sampling.SequentialDnaSampler import SequentialDNASampler

HEAD_ORDER = ["instrument", "velocity", "beat_unit", "offset", "grid_factor", "bpm", "time_signature"]


def main():
    ap = argparse.ArgumentParser(description="Inference demo for MultiTask DNA xLSTM (CUDA-only).")
    ap.add_argument("--config", type=str, default="experiment_multitask_relgu_finetune",
                    help="Path to the experiment YAML used for training. Defaults to Experiments/experiment_multitask_relgu_finetune.yml")
    ap.add_argument("--variations", type=int, default=1, help="Number of variations per DNA. Defaults to 1.")
    ap.add_argument("--max_tokens", type=int, default=200,)
    ap.add_argument("--model", type=str, default="GrooveModel",)
    ap.add_argument("--temperature", type=float, default=1.0,)
    ap.add_argument("--top_k", type=int, default=None,)
    ap.add_argument("--top_p", type=float, default=0.9,)
    ap.add_argument("--eval", type=bool, default=False)
    args = ap.parse_args()

    # --- Load training config via OmegaConf ---
    cfg_path = BASE_PATH / "Experiments" / f"{args.config}.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    # Paths from script location
    model_dir = BASE_PATH / 'Models'
    out_dir = BASE_PATH / 'Predictions'
    out_dir.mkdir(parents=True, exist_ok=True)

    dna_path = BASE_PATH / 'Data' / 'demo'
    if args.eval:
        dna_path = BASE_PATH / 'Data' / 'eval'

    dna_paths = get_all_dna_json_paths(dna_path)
    dnas = load_dna_json(dna_paths[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "multitask" in args.config:
        sampler = MultitaskDNASampler(cfg_path, model_dir, device, use_best=True)
    else:
        sampler = SequentialDNASampler(cfg_path, model_dir, device, use_best=True)

    for dna in tqdm(dnas, desc=f"Processing DNAs"):
        original, results = sampler.sample(
            dna,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            num_variations=args.variations,
        )

        # --- Save each DNA prediction to its own file ---
        for dna_entry in results:
            out_path = out_dir / f"{dna_entry['DNA_ID']}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([dna_entry], f, ensure_ascii=False, indent=2)
            print(f"[Saved] {out_path}")

        original_path = out_dir / f"{original['DNA_ID']}.json"
        with open(original_path, "w", encoding="utf-8") as f:
            json.dump([original], f, ensure_ascii=False, indent=2)
        print(f"[Saved] {original_path}")


if __name__ == "__main__":
    main()
