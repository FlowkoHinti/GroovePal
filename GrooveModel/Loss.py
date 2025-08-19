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
                weighted = torch.exp(-s) * base_loss + s
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
