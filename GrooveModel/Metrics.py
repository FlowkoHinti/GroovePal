from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Iterable

import torch
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import Perplexity


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
    Streaming metrics for multi-head DNA next-token tasks.

    Expected shapes:
      outputs: Dict[str, Tensor] where each tensor is (B, T, C) for a head.
      targets: Tensor of shape (B, T, H) where H == len(outputs).
               Channel i corresponds to the i-th (head, logits) pair iteration order.

    Supported metric names (in metric_names):
      - "accuracy"
      - "top_k_accuracy@K" (e.g., "top_k_accuracy@5")
      - "perplexity"

    Returns per-head keys "head/metric" and also "avg_<metric>" across heads.
    """

    def __init__(
        self,
        metric_names: Iterable[str],
        device: torch.device,
        ignore_index: Optional[int] = None,
    ):
        self.metric_names = list(metric_names)
        self.device = device
        self.ignore_index = ignore_index

        # Cache of torchmetrics instances: key -> Metric
        # key format: "head/metric_name"
        self._metrics: Dict[str, torch.nn.Module] = {}

    # ---- lifecycle ----
    def reset(self) -> None:
        for m in self._metrics.values():
            m.reset()

    # ---- internals ----
    def _ensure_metric(self, key: str, metric_name: str, num_classes: int):
        if key in self._metrics:
            return

        if metric_name.startswith("top_k_accuracy@"):
            k = int(metric_name.split("@", 1)[1])
            metric = MulticlassAccuracy(num_classes=num_classes, top_k=k)
        elif metric_name == "accuracy":
            metric = MulticlassAccuracy(num_classes=num_classes)
        elif metric_name == "perplexity":
            metric = Perplexity(ignore_index=self.ignore_index)
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

        self._metrics[key] = metric.to(self.device)

    # ---- streaming update ----
    @torch.no_grad()
    def update_batch(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> None:
        """
        outputs: {head: (B, T, C)}, targets: (B, T, H) matching the enumeration order of outputs.
        """
        for i, (head, logits) in enumerate(outputs.items()):
            num_classes = logits.size(-1)
            target_bt = targets[:, :, i]               # (B, T)
            logits_flat = logits.reshape(-1, num_classes)  # (B*T, C)
            target_flat = target_bt.reshape(-1)            # (B*T,)

            for metric_name in self.metric_names:
                key = f"{head}/{metric_name}"
                self._ensure_metric(key, metric_name, num_classes)
                m = self._metrics[key]

                if metric_name == "perplexity":
                    m.update(logits.to(self.device), target_bt.to(self.device))
                else:  # accuracy / top-k accuracy
                    m.update(logits_flat.to(self.device), target_flat.to(self.device))

    # ---- finalize ----
    @torch.no_grad()
    def compute_all(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        buckets: Dict[str, List[float]] = {}

        for key, metric in self._metrics.items():
            value = float(metric.compute())
            out[key] = value
            if "/" in key:
                _, metric_name = key.split("/", 1)
                buckets.setdefault(metric_name, []).append(value)

        return out