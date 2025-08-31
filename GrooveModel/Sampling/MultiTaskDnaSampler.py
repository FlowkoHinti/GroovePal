import logging
from os import PathLike
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple

import torch
from dacite import from_dict, Config as DaciteConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn

from Configs import BASE_PATH, MAX_SEQUENCE_LENGTH
from GrooveModel.Embedding.MultiTaskDnaEmbedding import MultiTaskDNAEmbeddingConfig
from GrooveModel.Models import ModelConfigxLstm, MultiTaskDNAxLSTM
from GrooveModel.Sampling.Sampler import DNATokenSampler
from GrooveModel.Tokenizer.MultiTaskDnaTokenizer import MultiTaskDnaTokenizer


# -----------------------------
# Utilities
# -----------------------------
def _compute_embedding_dim(embedding_cfg: DictConfig) -> int:
    # Matches Main.py's compute_embedding_dim
    return sum([
        embedding_cfg.instruments.embedding_dim,
        embedding_cfg.velocities.embedding_dim,
        embedding_cfg.offsets.embedding_dim,
        embedding_cfg.time_signature.embedding_dim,
        embedding_cfg.grid_factor.embedding_dim,
        embedding_cfg.bpm.embedding_dim,
        embedding_cfg.beat_units.embedding_dim,
    ])


def _inject_paths(cfg: DictConfig) -> DictConfig:
    # Matches Main.py's path resolution
    if isinstance(cfg.dataset.dna_path, str):
        cfg.dataset.dna_path = (BASE_PATH / cfg.dataset.dna_path).resolve()
    if isinstance(cfg.train.save_dir, str):
        cfg.train.save_dir = (BASE_PATH / cfg.train.save_dir).resolve()
    return cfg


def _prepare_config(cfg: DictConfig) -> DictConfig:
    # Align with Main.py behavior for MultiTask models
    cfg.model.context_length = MAX_SEQUENCE_LENGTH
    cfg.model.embedding_dim = _compute_embedding_dim(cfg.embedding)
    return _inject_paths(cfg)


def _load_cfg(cfg_or_path: Union[str, Path, DictConfig]) -> DictConfig:
    if isinstance(cfg_or_path, (str, Path)):
        cfg = OmegaConf.load(str(cfg_or_path))
    elif isinstance(cfg_or_path, DictConfig):
        cfg = cfg_or_path
    else:
        raise TypeError("cfg_or_path must be a path to YAML or an OmegaConf DictConfig.")
    return _prepare_config(cfg)


def _select_checkpoint_path(cfg: DictConfig, checkpoint: Optional[Union[str, Path]], use_best: bool) -> Path:
    if checkpoint is not None:
        p = Path(checkpoint)
        if p.is_file():
            return p
        elif p.is_dir():
            # Expect saved under <dir>/<model_name>/checkpoints/{best|latest}.pt
            base = p / cfg.train.model_name / "checkpoints"
        else:
            # treat as file path regardless
            return p
    else:
        base = Path(cfg.train.save_dir) / cfg.train.model_name / "checkpoints"

    ckpt = base / f"{cfg.train.model_name}_{'best' if use_best else 'latest'}.pt"
    if not ckpt.exists():
        # Fallback to the other one if preferred is missing
        alt = base / f"{cfg.train.model_name}_{'latest' if use_best else 'best'}.pt"
        if alt.exists():
            return alt
    return ckpt


def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / temperature


def _top_k_filter(logits: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    values, indices = torch.topk(logits, k, dim=-1)
    mask = torch.full_like(logits, float('-inf'))
    mask.scatter_(-1, indices, values)
    return mask


def _top_p_filter(logits: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    if top_p is None or top_p <= 0 or top_p >= 1:
        return logits
    # Sort by probability
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens outside nucleus
    cutoff = cumulative > top_p
    # Ensure at least one token remains
    cutoff[..., 0] = False

    sorted_probs = torch.where(cutoff, torch.zeros_like(sorted_probs), sorted_probs)
    # Map back to original index order
    new_logits = torch.full_like(logits, float('-inf'))
    new_logits.scatter_(-1, sorted_indices, torch.log(sorted_probs + 1e-20))
    return new_logits


def _categorical_sample(
    logits: torch.Tensor, temperature: float, top_k: Optional[int], top_p: Optional[float]
) -> Tuple[int, torch.Tensor]:
    """Return sampled index and probability distribution."""
    logits = _apply_temperature(logits, temperature)
    logits = _top_k_filter(logits, top_k)
    logits = _top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    ix = torch.multinomial(probs, num_samples=1).item()
    return ix, probs

# -----------------------------
# Multitask DNA Sampler
# -----------------------------
class MultitaskDNASampler(DNATokenSampler):
    """
    - Builds MultiTaskDNAxLSTM from a config (incl. absolute grid units via tokenizer cfg).
    - Restores weights from checkpoint (best or latest).
    - Tokenizes JSON DNA context, embeds -> forwards it.
    - Samples *the next token* heads with temperature/top-k/top-p.

    NOTE: We intentionally **do not** reconstruct the original DNA event.
    You said you'll take over that step using the sampled heads.
    """

    def __init__(
        self,
        cfg_or_path: Union[str, Path, DictConfig],
        checkpoint: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        use_best: bool = True,
    ):
        # Config
        self.cfg = _load_cfg(cfg_or_path)

        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build embedding + model exactly like in training
        self.embedding_config = from_dict(
            MultiTaskDNAEmbeddingConfig,
            OmegaConf.to_container(self.cfg.embedding, resolve=True),
            config=DaciteConfig(strict=True)
        )
        model_config = from_dict(
            ModelConfigxLstm,
            OmegaConf.to_container(self.cfg.model, resolve=True),
            config=DaciteConfig(strict=True)
        )
        self.model: nn.Module = MultiTaskDNAxLSTM(model_config, self.embedding_config)
        self.model.to(self.device)
        self.model.eval()

        # Restore weights
        ckpt_path = _select_checkpoint_path(self.cfg, checkpoint, use_best)
        state = torch.load(str(ckpt_path), map_location="cpu")
        self.model.load_state_dict(state["model_state_dict"], strict=True)

        # Set up tokenizer (critical for absolute grid units, etc.)
        self.tok_kwargs = dict(getattr(self.cfg.dataset, "tokenizer", {}))
        self.absolute_grid_units = self.tok_kwargs["absolute_grid_units"]
        self.tokenizer = MultiTaskDnaTokenizer()

    @torch.no_grad()
    def sample(
        self,
        dna_context: dict,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce ONE step of next-token predictions from the current context.
        If you pass max_tokens > 1 we will repeat this forward pass without
        feeding back sampled tokens (no teacher-forcing round-trip here) and
        return the *last* step.
        """
        steps = int(max_tokens or 1)

        # --- Tokenize context ---
        # tokens: (T, 7) LongTensor; beat_positions: (T,) LongTensor (or similar)
        tokens, beat_positions = self.tokenizer.tokenize(dna_context, **self.tok_kwargs)

        # Limit to model context
        T = tokens.size(0)
        ctx_len = min(T, self.cfg.model.context_length)
        tokens_ctx = tokens[-ctx_len:].unsqueeze(0).to(self.device)           # (1, ctx, 7)
        beat_ctx = beat_positions[-ctx_len:].unsqueeze(0).to(self.device)     # (1, ctx)

        out_payload: Dict[str, Any] = {}

        for _ in range(steps):
            class_logits, reg_outputs = self.model((tokens_ctx, beat_ctx))  # dicts of (B, T, C) / (B, T, 1)

            # Take last time step
            last_ix = -1
            sampled_ids: Dict[str, int] = {}
            probs_out: Dict[str, torch.Tensor] = {}

            for head_name in ("instrument", "beat_unit", "grid_factor", "bpm", "time_signature"):
                head_logit = class_logits[head_name][0, last_ix, :]  # (C,)
                idx, probs = _categorical_sample(head_logit, temperature, top_k, top_p)
                sampled_ids[head_name] = int(idx)
                probs_out[head_name] = probs.detach().cpu()

            # Regression heads: return normalized scalars as-is.
            vel = reg_outputs["velocity"][0, last_ix, 0].item()  # expected in [0,1]
            off = reg_outputs["offset"][0, last_ix, 0].item()    # expected in [-1,1]

            out_payload = {
                "class_ids": sampled_ids,
                "regression": {
                    "velocity_norm": float(vel),
                    "offset_norm": float(off),
                },
                "probs": probs_out,
            }

            # TODO:

            # Pull meta from the original song JSON (with safe defaults if missing)
            dna_id = dna_context.get("DNA_ID", f"...")
            bpm = dna_context.get("Bpm", None)
            numerator = dna_context.get("Numerator", None)
            denominator = dna_context.get("Denominator", None)
            grid_factor = dna_context.get("GridFactor", None)
            ticks_per_qn = dna_context.get("TicksPerQuarterNote", None)
            ticks_per_gu = dna_context.get("TicksPerGridUnit", None)

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
                bar_dna = [{'Value': 0, 'OffsetTicksPerValuePart': {}, 'VelocityPerValuePart': {}, 'IsEmpty': True}
                           for _ in
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

            # --- Run predictions and build requested output structure ---
            """
            output_payload = []
            for i, song in enumerate(all_songs):
                pred_tokens = predict_tokens_for_song(model, song, tokenizer_kwargs, device)

                

                

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
        """
        return out_payload