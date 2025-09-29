from typing import Dict, Optional

import torch
import torch.nn as nn


class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """
    Kendall et al., 'Multi-Task Learning Using Uncertainty to Weigh Losses' (CVPR 2018).

    Each task i has a learnable parameter s_i = log(sigma_i^2).
    We combine per-task losses as:
      - CE:  exp(-s_i) * CE + s_i
      - MSE: 0.5 * exp(-s_i) * MSE + 0.5 * s_i
      - MAE: exp(-0.5*s_i) * MAE + 0.5 * s_i

    Args:
        tasks: dict mapping task_name -> {"type": "ce"|"mse"|"mae"}.
               Example: {"semantic": {"type": "ce"}, "depth": {"type": "mae"}}
        reduction: "mean" or "sum" for the base losses (applied before weighting).
        init_log_vars: optional dict mapping task_name -> float initial s_i (log sigma^2).
                       Defaults to 0.0 (sigma=1) for all tasks.
        ce_kwargs: optional kwargs passed to nn.CrossEntropyLoss for CE tasks
                   (e.g., {"ignore_index":255, "label_smoothing":0.1, "weight":class_weights_tensor})
    """

    def __init__(
            self,
            tasks: Dict[str, Dict[str, str]],
            reduction: str = "mean",
            init_log_vars: Optional[Dict[str, float]] = None,
            ce_kwargs: Optional[Dict[str, dict]] = None,
            device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert reduction in ("mean", "sum"), "reduction must be 'mean' or 'sum'"
        self.task_specs = {}
        self.reduction = reduction
        self.ce_kwargs = ce_kwargs or {}
        self.device = device

        # Per-task uncertainty parameters s_i
        params = {}
        for name, spec in tasks.items():
            t = spec.get("type", "").lower()
            assert t in ("ce", "mse", "mae"), f"Unsupported task type: {t}"
            self.task_specs[name] = t
            s0 = 0.0 if init_log_vars is None else float(init_log_vars.get(name, 0.0))
            params[name] = nn.Parameter(torch.tensor([s0], device=device))
        self.log_vars = nn.ParameterDict(params)

        # Stateful base loss modules per task
        crit = {}
        for name, t in self.task_specs.items():
            if t == "ce":
                kwargs = dict(self.ce_kwargs.get(name, {}))  # copy to avoid side effects
                # Ensure reduction is set by this class
                kwargs["reduction"] = self.reduction
                crit[name] = nn.CrossEntropyLoss(**kwargs)
            elif t == "mse":
                crit[name] = nn.MSELoss(reduction=self.reduction)
            elif t == "mae":
                crit[name] = nn.L1Loss(reduction=self.reduction)
        self.criteria = nn.ModuleDict(crit)

    def forward(
            self,
            outputs: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            diagnostics: bool = False,
    ):
        """
        Args:
            outputs: dict task_name -> prediction tensor.
                     - For CE tasks: logits of shape (N, C, ...) matching CE requirements.
                     - For regression tasks (MSE/MAE): prediction tensor matching target shape.
            targets: dict task_name -> target tensor.
                     - For CE tasks: class indices (N, ...) or probabilities if using CE with soft targets.
                     - For regression tasks: same shape as outputs.
            diagnostics: Turns on/off the collection of task details for logging

        Returns:
            total_loss: scalar tensor
            details: dict with unweighted per-task losses, weighted terms, and current sigmas.
        """
        total = 0.0
        details = {"per_task": {}, "sigmas": {}} if diagnostics else None

        for name, loss_type in self.task_specs.items():
            s = self.log_vars[name]  # shape [1]
            base_loss = self.criteria[name](outputs[name], targets[name])

            if loss_type == "ce":
                weighted = torch.exp(-s) * base_loss + 0.5 * s
            elif loss_type == "mse":
                weighted = 0.5 * torch.exp(-s) * base_loss + 0.5 * s
            elif loss_type == "mae":
                # Laplacian likelihood: (1/sigma) * L1 + log sigma
                # with s = log(sigma^2) => sigma = exp(0.5*s)
                weighted = torch.exp(-0.5 * s) * base_loss + 0.5 * s
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            total = total + weighted

            if diagnostics:
                sigma = torch.sqrt(torch.exp(s))  # sigma = sqrt(exp(s)) since s = log sigma^2
                details["per_task"][name] = {
                    "base_loss": base_loss.detach(),
                    "weighted": weighted.detach(),
                    "s": s.detach(),
                }
                details["sigmas"][name] = sigma.detach()

        return total, details

    def get_sigmas(self) -> Dict[str, float]:
        """Return current sigma per task as plain floats (for logging)."""
        out = {}
        for name, s in self.log_vars.items():
            sigma = torch.sqrt(torch.exp(s)).item()
            out[name] = sigma
        return out

    def extra_repr(self) -> str:
        kinds = ", ".join(f"{k}:{v}" for k, v in self.task_specs.items())
        return f"tasks={{{kinds}}}, reduction={self.reduction}"


import torch
import torch.nn as nn
from typing import Dict, Optional


class ManualMultitaskLoss(nn.Module):
    """
    Combine multiple task losses with user-specified fixed weights.

    Args:
        tasks: dict mapping task_name -> {"type": "ce"|"mse"|"mae"}.
               Example: {"semantic": {"type": "ce"}, "depth": {"type": "mae"}}
        weights: dict mapping task_name -> float weight. If a task is missing,
                 it defaults to 1.0 unless strict_weights=True.
        reduction: "mean" or "sum" for the base losses (applied before weighting).
        ce_kwargs: optional kwargs passed to nn.CrossEntropyLoss for CE tasks
                   (e.g., {"ignore_index":255, "label_smoothing":0.1, "weight":class_weights_tensor})
        normalize: if True, re-normalize weights to sum to 1.0 at each forward (deterministic).
        strict_weights: if True, raise if any task weight is missing.
        device: torch.device for tensors inside this module (only affects CE class_weights tensors
                if provided here; model inputs still define actual device placement).
    """

    def __init__(
        self,
        tasks: Dict[str, Dict[str, str]],
        weights: Optional[Dict[str, float]] = None,
        reduction: str = "mean",
        ce_kwargs: Optional[Dict[str, dict]] = None,
        normalize: bool = False,
        strict_weights: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert reduction in ("mean", "sum"), "reduction must be 'mean' or 'sum'"
        self.task_specs: Dict[str, str] = {}
        self.reduction = reduction
        self.ce_kwargs = ce_kwargs or {}
        self.device = device
        self.normalize = normalize
        self.strict_weights = strict_weights

        # Validate & store task specs
        for name, spec in tasks.items():
            t = spec.get("type", "").lower()
            assert t in ("ce", "mse", "mae"), f"Unsupported task type: {t}"
            self.task_specs[name] = t

        # Fixed weights
        weights = dict(weights or {})
        if strict_weights:
            missing = [k for k in self.task_specs if k not in weights]
            if missing:
                raise ValueError(f"Missing weights for tasks: {missing}")
        # Default to 1.0 for any missing tasks
        for k in self.task_specs:
            weights.setdefault(k, 1.0)

        # Basic sanity checks
        for k, w in weights.items():
            if k not in self.task_specs:
                raise ValueError(f"Weight provided for unknown task '{k}'")
            if not isinstance(w, (int, float)):
                raise TypeError(f"Weight for task '{k}' must be a number, got {type(w)}")
            if w < 0:
                raise ValueError(f"Weight for task '{k}' must be non-negative, got {w}")

        # Keep weights as a buffer (non-trainable) for easy state_dict I/O
        self.register_buffer(
            "weights_tensor",
            torch.tensor([weights[k] for k in self.task_specs.keys()], dtype=torch.float32),
            persistent=True,
        )
        self.task_names = list(self.task_specs.keys())  # stable ordering

        # Stateful base loss modules per task
        crit = {}
        for name, t in self.task_specs.items():
            if t == "ce":
                kwargs = dict(self.ce_kwargs.get(name, {}))
                kwargs["reduction"] = self.reduction
                # If CE class weights tensor is provided here and a device is set, move it
                if "weight" in kwargs and isinstance(kwargs["weight"], torch.Tensor) and device is not None:
                    kwargs["weight"] = kwargs["weight"].to(device)
                crit[name] = nn.CrossEntropyLoss(**kwargs)
            elif t == "mse":
                crit[name] = nn.MSELoss(reduction=self.reduction)
            elif t == "mae":
                crit[name] = nn.L1Loss(reduction=self.reduction)
        self.criteria = nn.ModuleDict(crit)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        diagnostics: bool = False,
    ):
        """
        Args:
            outputs: dict task_name -> prediction tensor.
                     - For CE tasks: logits of shape (N, C, ...) matching CE requirements.
                     - For regression tasks (MSE/MAE): prediction tensor matching target shape.
            targets: dict task_name -> target tensor.
                     - For CE tasks: class indices (N, ...) or probabilities if using CE with soft targets.
                     - For regression tasks: same shape as outputs.
            diagnostics: if True, returns per-task base losses and weighted terms.

        Returns:
            total_loss: scalar tensor
            details: dict with per-task base losses, weights, and weighted losses (if diagnostics)
        """
        # Determine weights (optionally normalized each call)
        w = self.weights_tensor
        if self.normalize:
            denom = torch.clamp(w.sum(), min=torch.finfo(w.dtype).eps)
            w = w / denom

        total = 0.0
        details = {"per_task": {}, "weights": {}} if diagnostics else None

        for idx, name in enumerate(self.task_names):
            loss_type = self.task_specs[name]
            base_loss = self.criteria[name](outputs[name], targets[name])
            weighted = w[idx] * base_loss
            total = total + weighted

            if diagnostics:
                details["per_task"][name] = {
                    "base_loss": base_loss.detach(),
                    "weighted": weighted.detach(),
                }
                details["weights"][name] = w[idx].detach()

        return total, details

    def get_weights(self) -> Dict[str, float]:
        """Return current fixed weights as plain floats in task order."""
        return {name: float(self.weights_tensor[i].item()) for i, name in enumerate(self.task_names)}

    def set_weights(self, new_weights: Dict[str, float], normalize: Optional[bool] = None):
        """
        Update weights at runtime. Optionally re-normalize after setting.
        """
        for i, name in enumerate(self.task_names):
            if name in new_weights:
                val = float(new_weights[name])
                if val < 0:
                    raise ValueError(f"Weight for task '{name}' must be non-negative, got {val}")
                self.weights_tensor[i] = val
        if normalize is None:
            normalize = self.normalize
        if normalize:
            denom = torch.clamp(self.weights_tensor.sum(), min=torch.finfo(self.weights_tensor.dtype).eps)
            self.weights_tensor[:] = self.weights_tensor / denom

    def extra_repr(self) -> str:
        kinds = ", ".join(f"{k}:{v}" for k, v in self.task_specs.items())
        ws = ", ".join(f"{k}:{self.weights_tensor[i].item():.4g}" for i, k in enumerate(self.task_names))
        norm = "True" if self.normalize else "False"
        return f"tasks={{{kinds}}}, reduction={self.reduction}, weights={{{ws}}}, normalize={norm}"


class NaiveMultitaskLoss(nn.Module):
    """
    Naive multi-task loss: just sums all task losses equally.

    Each task contributes its base loss directly:
        total_loss = sum_i base_loss_i

    Args:
        tasks: dict mapping task_name -> {"type": "ce"|"mse"|"mae"}.
               Example: {"semantic": {"type": "ce"}, "depth": {"type": "mae"}}
        reduction: "mean" or "sum" for the base losses (applied before summing).
        ce_kwargs: optional dict of per-task kwargs for CE tasks.
                   e.g. {"semantic": {"ignore_index":255}}
    """

    def __init__(
        self,
        tasks: Dict[str, Dict[str, str]],
        reduction: str = "mean",
        ce_kwargs: Optional[Dict[str, dict]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert reduction in ("mean", "sum"), "reduction must be 'mean' or 'sum'"
        self.task_specs: Dict[str, str] = {}
        self.reduction = reduction
        self.ce_kwargs = ce_kwargs or {}
        self.device = device

        # validate & register task specs
        for name, spec in tasks.items():
            t = spec.get("type", "").lower()
            assert t in ("ce", "mse", "mae"), f"Unsupported task type: {t}"
            self.task_specs[name] = t

        # per-task loss modules
        crit = {}
        for name, t in self.task_specs.items():
            if t == "ce":
                kwargs = dict(self.ce_kwargs.get(name, {}))
                kwargs["reduction"] = self.reduction
                if "weight" in kwargs and isinstance(kwargs["weight"], torch.Tensor) and device is not None:
                    kwargs["weight"] = kwargs["weight"].to(device)
                crit[name] = nn.CrossEntropyLoss(**kwargs)
            elif t == "mse":
                crit[name] = nn.MSELoss(reduction=self.reduction)
            elif t == "mae":
                crit[name] = nn.L1Loss(reduction=self.reduction)
        self.criteria = nn.ModuleDict(crit)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        diagnostics: bool = False,
    ):
        """
        Args:
            outputs: dict task_name -> model prediction
            targets: dict task_name -> target tensor
            diagnostics: if True, returns per-task losses

        Returns:
            total_loss: scalar tensor
            details (optional): dict with per-task base losses
        """
        total = 0.0
        details = {"per_task": {}} if diagnostics else None

        for name, loss_fn in self.criteria.items():
            base_loss = loss_fn(outputs[name], targets[name])
            total = total + base_loss

            if diagnostics:
                details["per_task"][name] = {"base_loss": base_loss.detach()}

        return total, details

    def extra_repr(self) -> str:
        kinds = ", ".join(f"{k}:{v}" for k, v in self.task_specs.items())
        return f"tasks={{{kinds}}}, reduction={self.reduction}"