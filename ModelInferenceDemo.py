import argparse
import json
from pathlib import Path

import torch
from dacite import from_dict, Config as DaciteConfig
from omegaconf import OmegaConf, DictConfig

from Configs import BASE_PATH, MAX_SEQUENCE_LENGTH
from GrooveModel.Datasets import get_all_dna_json_paths, load_dna_json
from GrooveModel.Embedding.Embedding import MultiTaskDNAEmbeddingConfig
from GrooveModel.Models import MultiTaskDNAxLSTM, ModelConfigxLstm
from GrooveModel.Tokenizer.Tokenizer import MultiTaskDnaTokenizer, MultiDnaToken, tokens_to_tensor
from GrooveModel.Utils.BeatUnit import decode_beat_unit
from GrooveModel.Utils.DNAOffset import decode_offset_ticks
from GrooveModel.Utils.DNAValue import decode_instrument
from GrooveModel.Utils.DNAVelocity import decode_velocity

HEAD_ORDER = ["instrument", "velocity", "beat_unit", "offset", "grid_factor", "bpm", "time_signature"]


def build_model_from_config(cfg: DictConfig, device: torch.device) -> MultiTaskDNAxLSTM:
    embedding_config = from_dict(
        MultiTaskDNAEmbeddingConfig,
        OmegaConf.to_container(cfg.embedding, resolve=True),
        config=DaciteConfig(strict=True)
    )

    model_config = from_dict(
        ModelConfigxLstm,
        OmegaConf.to_container(cfg.model, resolve=True),
        config=DaciteConfig(strict=True)
    )

    model = MultiTaskDNAxLSTM(model_config, embedding_config)
    model.reset_parameters()
    model.to(device)
    model.eval()
    return model


def load_best_checkpoint(model: torch.nn.Module, cfg: DictConfig, device: torch.device) -> Path:
    model_name: str = cfg.train.model_name
    save_dir = cfg.train.save_dir
    # ensure Path
    save_dir = (BASE_PATH / save_dir).resolve() if isinstance(save_dir, str) else Path(save_dir)
    ckpt_dir = save_dir / model_name / "checkpoints"
    best_path = ckpt_dir / f"{model_name}_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found at: {best_path}")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return best_path


@torch.no_grad()
def predict_tokens_for_song(model: MultiTaskDNAxLSTM, song_json: dict, tokenizer_kwargs: dict, device: torch.device):
    # Tokenize
    tokens = MultiTaskDnaTokenizer.tokenize(song_json, **tokenizer_kwargs)
    if len(tokens) < 2:
        return []  # nothing to predict

    # Next-token style: input is everything except the last step
    input_tokens = tokens[:-1]
    x = tokens_to_tensor(input_tokens).unsqueeze(0).to(device)  # [1, T, 7]

    # Forward
    logits = model(x)  # dict(head -> [1, T, vocab])

    # Argmax per head
    preds_per_head = {}
    for head in HEAD_ORDER:
        head_logits = logits[head]  # [1, T, vocab]
        head_ids = head_logits.argmax(dim=-1).squeeze(0).tolist()  # [T]
        preds_per_head[head] = head_ids

    # Recombine heads -> one MultiDnaToken per timestep
    out = []
    T = len(next(iter(preds_per_head.values()))) if preds_per_head else 0
    for t in range(T):
        token = MultiDnaToken(
            Instrument=preds_per_head["instrument"][t],
            Velocity=preds_per_head["velocity"][t],
            BeatUnit=preds_per_head["beat_unit"][t],
            BeatUnitOffset=preds_per_head["offset"][t],
            GridFactor=preds_per_head["grid_factor"][t],
            Bpm=preds_per_head["bpm"][t],
            TimeSignature=preds_per_head["time_signature"][t],
        )
        out.append(token.__dict__)  # serialize as raw integer fields
    return out


def compute_embedding_dim(embedding_cfg: DictConfig) -> int:
    # Keep everything in OmegaConf attribute style for clarity.
    return sum([
        int(embedding_cfg.instruments.embedding_dim),
        int(embedding_cfg.velocities.embedding_dim),
        int(embedding_cfg.offsets.embedding_dim),
        int(embedding_cfg.time_signature.embedding_dim),
        int(embedding_cfg.grid_factor.embedding_dim),
        int(embedding_cfg.bpm.embedding_dim),
        int(embedding_cfg.beat_units.embedding_dim),
    ])


def inject_paths(cfg: DictConfig) -> DictConfig:
    if isinstance(cfg.train.save_dir, str):
        cfg.train.save_dir = (BASE_PATH / cfg.train.save_dir).resolve()
    return cfg


def prepare_config(cfg: DictConfig) -> DictConfig:
    """Inject dynamic values like embedding_dim and context_length using OmegaConf."""
    cfg.model.context_length = int(MAX_SEQUENCE_LENGTH)
    cfg.model.embedding_dim = compute_embedding_dim(cfg.embedding)
    cfg = inject_paths(cfg)
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Inference demo for MultiTask DNA xLSTM (CUDA-only).")
    ap.add_argument("--config", type=str, default="experiment_initial.yml",
                    help="Path to the experiment YAML used for training. Defaults to Experiments/experiment_initial.yml")
    args = ap.parse_args()

    # --- Load training config via OmegaConf ---
    cfg_path = Path("Experiments") / args.config
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg: DictConfig = OmegaConf.load(str(cfg_path))

    # --- Enforce CUDA (your xLSTM cells are GPU-only) ---
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Please run on a GPU machine.")
    device = torch.device("cuda")

    # --- Prepare config, build model, and load best checkpoint ---
    cfg = prepare_config(cfg)
    model = build_model_from_config(cfg, device)
    best_path = load_best_checkpoint(model, cfg, device)
    print(f"[Info] Using checkpoint: {best_path}")

    # Paths from script location
    data_root = Path(__file__).resolve().parent  # e.g., repository root containing Data/
    demo_dir = data_root / 'Data' / 'demo'
    out_dir = data_root / "Predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Gather and concatenate all demo songs across all *.json ---
    demo_json_paths = get_all_dna_json_paths(str(demo_dir))
    if not demo_json_paths:
        raise FileNotFoundError(f"No .json files found in {demo_dir}")

    print(f"[Info] Found {len(demo_json_paths)} demo file(s) in {demo_dir}")
    all_songs = []
    for p in sorted(demo_json_paths):
        songs = load_dna_json(p)  # Expect list of song dicts
        if isinstance(songs, list):
            all_songs.extend(songs)
        else:
            raise ValueError(f"Expected list of song dicts in {p}, got {type(songs)}")

    print(f"[Info] Total concatenated songs: {len(all_songs)}")

    # Tokenizer runtime kwargs (same keys as in your dataset config)
    tok_cfg = OmegaConf.select(cfg, "dataset.tokenizer", default={}) or {}
    # If tok_cfg is a DictConfig, convert to a dict for .get(...) ergonomics
    if isinstance(tok_cfg, DictConfig):
        tok_cfg = OmegaConf.to_container(tok_cfg, resolve=True) or {}

    tokenizer_kwargs = {
        "trim_leading_empty_measures": tok_cfg.get("trim_leading_empty_measures", True),
        "absolute_grid_units": tok_cfg.get("absolute_grid_units", False),
    }

    # --- Run predictions and build requested output structure ---
    output_payload = []
    for i, song in enumerate(all_songs):
        pred_tokens = predict_tokens_for_song(model, song, tokenizer_kwargs, device)

        # Pull meta from the original song JSON (with safe defaults if missing)
        dna_id = song.get("DNA_ID", f"song_{i}")
        bpm = song.get("Bpm", None)
        numerator = song.get("Numerator", None)
        denominator = song.get("Denominator", None)
        grid_factor = song.get("GridFactor", None)
        ticks_per_qn = song.get("TicksPerQuarterNote", None)
        ticks_per_gu = song.get("TicksPerGridUnit", None)

        # --- Find all measures (for relative gridunits) ---

        bar_indexes = []
        prev_were_one = False
        for i, token in enumerate(pred_tokens):
            if token['BeatUnit'] == 1 and not prev_were_one:
                bar_indexes.append(i)
                prev_were_one = True
            elif prev_were_one and token['BeatUnit'] != 1:
                prev_were_one = False

        bar_count = len(bar_indexes)

        # --- Extract DNA units per Bar ---

        bars = []
        for i, bar_index in enumerate(bar_indexes):
            if i < bar_count - 1:
                bars.append(pred_tokens[bar_index:bar_indexes[i + 1]])

        dna_units = []

        for bar in bars:
            bar_dna = [{'Value': 0, 'OffsetTicksPerValuePart': {}, 'VelocityPerValuePart': {}, 'IsEmpty': True} for _ in
                       range(numerator * grid_factor)]
            for token in bar:

                beat_unit = decode_beat_unit(token['BeatUnit'])
                value = decode_instrument(token['Instrument'])
                if value != 0:
                    velocity = decode_velocity(token['Velocity'])
                    offset = decode_offset_ticks(token['BeatUnitOffset'], ticks_per_grid_unit=ticks_per_gu)

                    dna_unit = bar_dna[beat_unit]
                    dna_unit['Value'] += value
                    dna_unit['OffsetTicksPerValuePart'][str(value)] = offset
                    dna_unit['VelocityPerValuePart'][str(value)] = velocity
                    dna_unit['IsEmpty'] = False

            dna_units.extend(bar_dna)

        output_payload.append({
            "DNA_ID": f"{dna_id}_pred",
            "DNASet": "demo",
            "AuthorData": cfg.train.model_name,
            "Bpm": bpm,
            "Numerator": numerator,
            "Denominator": denominator,
            "GridFactor": grid_factor,
            "TicksPerQuarterNote": ticks_per_qn,
            "TicksPerGridUnit": ticks_per_gu,
            "NumberOfBars": bar_count,
            "DNAUnits": dna_units,
        })

    # --- Save combined predictions to <Data>/Predictions/predictions.json ---
    out_path = out_dir / "predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {out_path}")

# TODO UPDATE TO NEW MODEL OUTPUT HEAD AND ADD SEQUENTIAL
if __name__ == "__main__":
    main()
