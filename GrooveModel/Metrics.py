from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Iterable

import torch
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import Perplexity

from GrooveModel.Utils.DNAOffset import OFFSET_TICKS_RESOLUTION
from GrooveModel.Utils.DNAVelocity import EFFECTIVE_VELOCITY_RESOLUTION
from GrooveModel.Utils.SpecialTokens import SpecialTokens, SPECIAL_TOKEN_SIZE


class BaseMetrics(ABC):
    """
    Minimal interface for streaming metrics over one evaluation epoch.
    Implementations should:
      - cache metric objects,
      - support reset() per epoch,
      - update once per batch,
      - compute_all() at epoch end.
    """

    @abstractmethod
    def reset(self) -> None:
        ...

    @torch.no_grad()
    @abstractmethod
    def update_batch(self, outputs, targets) -> None:
        """
        Update all metrics with a single batch.
        Shapes/types are defined by the concrete implementation.
        """
        ...

    @torch.no_grad()
    @abstractmethod
    def compute_all(self) -> Dict[str, float]:
        """
        Return a flat dict of metrics. For multi-head cases, keys like 'head/metric'
        are recommended. Implementations may also add 'avg_<metric>' aggregations.
        """
        ...


class MultiTaskDNAMetrics(BaseMetrics):
    """
    Streaming metrics for multi-head DNA learner.

    Classification heads (logits): accuracy, top_k_accuracy@K, perplexity

    Regression heads (preds): mse, mae

    `update_batch` accepts:
      outputs:
        - Dict[str, (B,T,C)]  (cls-only), or
        - Tuple[Dict[str,(B,T,C)], Dict[str,(B,T,1 or B,T)]]  -> (class_logits, reg_outputs)
      targets: (B, T, H_total) token ids in the fixed order below.
    """

    def __init__(
        self,
        metric_names: Iterable[str],
        device: torch.device,
        ignore_index: Optional[int] = None,
        head_index: Optional[Dict[str, int]] = None,
        regression_heads: Optional[List[str]] = None,
    ):
        self.metric_names = list(metric_names)
        self.device = device
        self.ignore_index = ignore_index

        # Indices in targets' last dim (adjust if your layout differs)
        self.head_index = head_index or {
            "instrument": 0,
            "velocity": 1,       # regression (tokenized)
            "beat_unit": 2,
            "offset": 3,         # regression (tokenized)
            "grid_factor": 4,
            "bpm": 5,
            "time_signature": 6,
        }
        self.regression_heads = set(regression_heads or ["velocity", "offset"])
        self._metrics: Dict[str, torch.nn.Module] = {}

    # ---- lifecycle ----
    def reset(self) -> None:
        for m in self._metrics.values():
            m.reset()

    # ---- metric factories ----
    def _ensure_cls_metric(self, key: str, name: str, num_classes: int):
        if key in self._metrics:
            return
        if name.startswith("top_k_accuracy@"):
            k = int(name.split("@", 1)[1])
            metric = MulticlassAccuracy(num_classes=num_classes, top_k=k, ignore_index=self.ignore_index)
        elif name == "accuracy":
            metric = MulticlassAccuracy(num_classes=num_classes, ignore_index=self.ignore_index)
        elif name == "perplexity":
            metric = Perplexity(ignore_index=self.ignore_index)
        else:
            raise ValueError(f"Unsupported classification metric: {name}")
        self._metrics[key] = metric.to(self.device)

    def _ensure_reg_metric(self, key: str, name: str):
        if key in self._metrics:
            return
        if name == "mse":
            metric = MeanSquaredError()
        elif name == "mae":
            metric = MeanAbsoluteError()
        else:
            raise ValueError(f"Unsupported regression metric: {name}")
        self._metrics[key] = metric.to(self.device)

    # ---- fast target decoders ----
    @staticmethod
    def _decode_velocity_tokens(token_ids: torch.Tensor) -> torch.Tensor:
        # token -> [0,1], PAD handled by mask outside
        idx = (token_ids.long() - SPECIAL_TOKEN_SIZE).clamp(0, EFFECTIVE_VELOCITY_RESOLUTION - 1)
        denom = max(1, EFFECTIVE_VELOCITY_RESOLUTION - 1)
        return idx.to(torch.float32) / float(denom)

    @staticmethod
    def _decode_offset_tokens(token_ids: torch.Tensor) -> torch.Tensor:
        # token -> [-1,1], PAD handled by mask outside
        idx = (token_ids.long() - SPECIAL_TOKEN_SIZE).clamp(0, OFFSET_TICKS_RESOLUTION - 1)
        denom = max(1, OFFSET_TICKS_RESOLUTION - 1)
        z01 = idx.to(torch.float32) / float(denom)
        return 2.0 * z01 - 1.0

    # ---- streaming update ----
    @torch.no_grad()
    def update_batch(self, outputs, targets: torch.Tensor) -> None:
        """
        Pass the raw model outputs (tuple or dict) and the full target tensor.
        The model should already apply sigmoid/tanh to regression outputs.
        """
        # Unpack possible tuple
        if isinstance(outputs, tuple):
            class_logits, reg_outputs = outputs[0], (outputs[1] or {})
        else:
            class_logits, reg_outputs = outputs, {}

        # ---- classification metrics ----
        if isinstance(class_logits, dict):
            for head, logits in class_logits.items():
                if head in self.regression_heads:
                    continue
                num_classes = logits.size(-1)
                idx = self.head_index[head]
                tgt_bt = targets[:, :, idx]                   # (B,T)

                # Flatten for accuracy; perplexity can use (B,T,C)
                logits_flat = logits.reshape(-1, num_classes)
                tgt_flat = tgt_bt.reshape(-1)

                for name in self.metric_names:
                    if name in ("mse", "mae"):
                        continue
                    key = f"{head}/{name}"
                    self._ensure_cls_metric(key, name, num_classes)
                    m = self._metrics[key]
                    if name == "perplexity":
                        m.update(logits.to(self.device), tgt_bt.to(self.device))
                    else:
                        m.update(logits_flat.to(self.device), tgt_flat.to(self.device))

        # ---- regression metrics (velocity, offset, …) ----
        if isinstance(reg_outputs, dict):
            for head, pred in reg_outputs.items():
                if head not in self.regression_heads:
                    continue

                # Predictions already bounded by the model; shape (B,T,1) or (B,T)
                y_pred = pred.squeeze(-1)

                if head == "velocity":
                    tgt_ids = targets[:, :, self.head_index["velocity"]]
                    mask = (tgt_ids != SpecialTokens.PAD)
                    y_true = self._decode_velocity_tokens(tgt_ids)
                elif head == "offset":
                    tgt_ids = targets[:, :, self.head_index["offset"]]
                    mask = (tgt_ids != SpecialTokens.PAD)
                    y_true = self._decode_offset_tokens(tgt_ids)
                else:
                    # Generic fallback (if you later add continuous targets directly)
                    tgt_ids = targets[:, :, self.head_index[head]]
                    mask = (tgt_ids != SpecialTokens.PAD)
                    y_true = tgt_ids.to(torch.float32)

                if mask.any():
                    yp = y_pred[mask].reshape(-1).to(self.device)
                    yt = y_true[mask].reshape(-1).to(self.device)
                    for name in self.metric_names:
                        if name not in ("mse", "mae"):
                            continue
                        key = f"{head}/{name}"
                        self._ensure_reg_metric(key, name)
                        self._metrics[key].update(yp, yt)

    # ---- finalize ----
    @torch.no_grad()
    def compute_all(self) -> Dict[str, float]:
        return {k: float(m.compute()) for k, m in self._metrics.items()}