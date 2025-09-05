# top k/ top p
# temperature
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional, Any, Union, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from Configs import BASE_PATH, MAX_SEQUENCE_LENGTH


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

    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Keep the minimal set whose cumulative sum >= top_p
    keep = (cumulative - sorted_probs) < top_p
    keep[..., 0] = True  # ensure at least one token

    kept_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))

    # Convert kept probs back to logits; others to -inf
    new_logits = torch.full_like(logits, float('-inf'))
    new_logits.scatter_(-1, sorted_indices, torch.log(kept_probs + 1e-20))
    return new_logits


def _categorical_sample(
        logits: torch.Tensor, temperature: float, top_k: Optional[int], top_p: Optional[float]
) -> Tuple[int, torch.Tensor]:
    """Return sampled index and probability distribution."""
    logits = _apply_temperature(logits, temperature)
    logits = _top_k_filter(logits, top_k)
    logits = _top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    probs = probs.squeeze()
    ix = torch.multinomial(probs, num_samples=1).item()
    return ix, probs


class DNATokenSampler(ABC):
    """
    Abstract base class for sampling DNA token sequences from generative models.

    Provides a consistent interface for sampling with temperature,
    top-k, top-p (nucleus), etc.
    """

    @abstractmethod
    def sample(
            self,
            dna_context: dict,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs: dict[str, Any],
    ) -> dict:
        """
        Generate a DNA token sequence given a context.

        Args:
            dna_context (dict): List of input tokens (DNA-json).
            temperature (float, optional): Softmax temperature > 0.
            top_k (int, optional): Keep only the top-k most likely tokens.
            top_p (float, optional): Nucleus sampling cutoff (prob. mass).
            max_tokens (int, optional): Maximum number of tokens to generate.
            **kwargs: Extra model-specific parameters.

        Returns:
            List[str]: Generated sequence of DNA tokens.
        """
        pass
