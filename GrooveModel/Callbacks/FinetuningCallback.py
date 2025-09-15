# https://ijisrt.com/assets/upload/files/IJISRT20NOV654.pdf -> quadrant 3
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn

from GrooveModel.Callbacks.Callback import Callback

try:
    from GrooveModel.xlstm.xlstm.xlstm_block_stack import xLSTMBlockStack as _XLSTM_CLS
except Exception:
    _XLSTM_CLS = None


class FineTuneCallback(Callback):
    """
    1) Optionally loads weights from a previous model checkpoint/state_dict.
    2) Freezes all weights except output heads at the start of training.
    3) Gradually unfreezes remaining layers from latter (deeper) to earlier ones.
       Includes per-block grouping for xLSTMBlockStack.
    """

    def __init__(
            self,
            enabled: bool = False,
            init_from: Optional[str] = None,
            start_epoch: int = 1,
            unfreeze_per_epoch: int = 1,
            strict_loading: bool = False,
            logger=None,
    ):
        super().__init__(logger=logger)
        self.enabled = enabled
        self.init_from = Path(init_from) if init_from else None
        self.start_epoch = int(start_epoch)
        self.unfreeze_per_epoch = max(1, int(unfreeze_per_epoch))
        self.strict_loading = bool(strict_loading)

        self._model: Optional[nn.Module] = None
        self._optimizer = None

        self._body_groups: List[List[Tuple[str, nn.Parameter]]] = []
        self._head_params: List[Tuple[str, nn.Parameter]] = []

        self._n_groups_total: int = 0
        self._n_groups_unfrozen: int = 0

    def on_train_begin(self, learner_state, **kwargs):
        if not self.enabled:
            return

        self._model = learner_state.model
        self._optimizer = learner_state.optimizer

        if self.init_from is not None:
            self._load_from_checkpoint(self.init_from, self.strict_loading)

        self._head_params, self._body_groups = self._group_parameters(self._model)
        self._n_groups_total = len(self._body_groups)
        self._n_groups_unfrozen = 0

        self._apply_freezing()
        self._log_param_overview("on_train_begin")

    def on_epoch_begin(self, learner_state, **kwargs):
        if not self.enabled or self._model is None:
            return

        epoch = int(learner_state.epoch)
        if epoch >= self.start_epoch and self._n_groups_unfrozen < self._n_groups_total:
            remaining = self._n_groups_total - self._n_groups_unfrozen
            n_this_epoch = min(self.unfreeze_per_epoch, remaining)

            start_idx = self._n_groups_total - self._n_groups_unfrozen - n_this_epoch
            end_idx = self._n_groups_total - self._n_groups_unfrozen

            for gi in range(start_idx, end_idx):
                for _, p in self._body_groups[gi]:
                    p.requires_grad = True

            self._n_groups_unfrozen += n_this_epoch
            self.logger.info(
                f"[FineTune] Epoch {epoch}: unfroze {n_this_epoch} group(s) "
                f"(total {self._n_groups_unfrozen}/{self._n_groups_total})."
            )
            self._refresh_optimizer_param_groups()

    def _load_from_checkpoint(self, path: Path, strict: bool):
        try:
            ckpt = torch.load(path, map_location="cpu")
            state_dict = ckpt["model_state_dict"]

            missing, unexpected = self._model.load_state_dict(state_dict, strict=strict)
            if strict:
                self.logger.info("[FineTune] Loaded checkpoint strictly.")
            else:
                self.logger.info(
                    f"[FineTune] Loaded checkpoint non-strictly "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})."
                )
                if missing:
                    self.logger.debug(f"[FineTune] Missing keys (first 20): {missing[:20]}")
                if unexpected:
                    self.logger.debug(f"[FineTune] Unexpected keys (first 20): {unexpected[:20]}")
        except Exception as e:
            self.logger.error(f"[FineTune] Failed to load weights from '{path}': {e}")

    @staticmethod
    def _looks_like_head(name: str, module_path: str, tie_weights: bool) -> bool:
        name_l = name.lower()
        path_l = module_path.lower()
        HEAD_TOKENS = [
            "head", "heads",  # generic
            "classifier", "classification",
            "regression",
            "output_head",  # sequential model
        ]
        if tie_weights:
            HEAD_TOKENS.append("embedding")

        return any(k in name_l for k in HEAD_TOKENS) or any(k in path_l for k in HEAD_TOKENS)

    @classmethod
    def _is_xlstm_stack(cls, mod: nn.Module, name: str) -> bool:
        if _XLSTM_CLS is not None and isinstance(mod, _XLSTM_CLS):
            return True
        if hasattr(mod, "blocks") and isinstance(getattr(mod, "blocks"), nn.ModuleList):
            lname = name.lower()
            if any(tag in lname for tag in ("xlstm", "stack")):
                return True
        return False

    def _group_parameters(
            self, model: nn.Module
    ) -> Tuple[List[Tuple[str, nn.Parameter]], List[List[Tuple[str, nn.Parameter]]]]:
        head_params: List[Tuple[str, nn.Parameter]] = []
        body_groups: List[List[Tuple[str, nn.Parameter]]] = []

        for n, p in model.named_parameters():
            mod_path = n.rsplit(".", 1)[0] if "." in n else n
            if self._looks_like_head(n, mod_path, model.model_config.tie_weights):
                head_params.append((n, p))

        def params_under(prefix: str) -> List[Tuple[str, nn.Parameter]]:
            items: List[Tuple[str, nn.Parameter]] = []
            pfx = prefix + "."
            for n, p in model.named_parameters():
                if n.startswith(pfx) and not self._looks_like_head(n, prefix, model.model_config.tie_weights):
                    items.append((n, p))
            return items

        def add_groups(prefix: str, mod: nn.Module):
            if self._is_xlstm_stack(mod, prefix):
                if hasattr(mod, "blocks") and isinstance(mod.blocks, nn.ModuleList):
                    for i, _blk in enumerate(mod.blocks):
                        ps = params_under(f"{prefix}.blocks.{i}")
                        if ps:
                            body_groups.append(ps)
                if hasattr(mod, "post_blocks_norm") and not isinstance(mod.post_blocks_norm, nn.Identity):
                    psn = params_under(f"{prefix}.post_blocks_norm")
                    if psn:
                        body_groups.append(psn)
                return

            if isinstance(mod, (nn.Sequential, nn.ModuleList)):
                for i, _child in enumerate(mod):
                    ps = params_under(f"{prefix}.{i}")
                    if ps:
                        body_groups.append(ps)
                return

            split = False
            for cname, child in mod.named_children():
                if self._looks_like_head(cname, f"{prefix}.{cname}", model.model_config.tie_weights):
                    continue
                ps = params_under(f"{prefix}.{cname}")
                if ps:
                    body_groups.append(ps)
                    split = True
            if split:
                return

            ps_self = params_under(prefix)
            if ps_self:
                body_groups.append(ps_self)

        for cname, child in model.named_children():
            if self._looks_like_head(cname, cname, model.model_config.tie_weights):
                continue
            add_groups(cname, child)

        assigned = {n for group in body_groups for (n, _) in group} | {n for (n, _) in head_params}
        orphan_params = [(n, p) for n, p in model.named_parameters() if n not in assigned]
        if orphan_params:
            body_groups.insert(0, orphan_params)

        return head_params, body_groups

    def _apply_freezing(self):
        for _, p in self._model.named_parameters():
            p.requires_grad = False
        enabled = 0
        for _, p in self._head_params:
            p.requires_grad = True
            enabled += 1
        self.logger.info(f"[FineTune] Initially froze model; enabled {enabled} head parameter tensors.")
        self._refresh_optimizer_param_groups()

    def _refresh_optimizer_param_groups(self):
        if self._optimizer is None:
            return
        try:
            params_with_grad = [p for p in self._model.parameters() if p.requires_grad]
            if hasattr(self._optimizer, "param_groups") and len(self._optimizer.param_groups) == 1:
                self._optimizer.param_groups[0]["params"] = params_with_grad
            else:
                for g in self._optimizer.param_groups:
                    g["params"] = [p for p in g.get("params", []) if getattr(p, "requires_grad", False)]
                present = {id(p) for g in self._optimizer.param_groups for p in g.get("params", [])}
                missing = [p for p in params_with_grad if id(p) not in present]
                if missing and self._optimizer.param_groups:
                    self._optimizer.param_groups[0]["params"].extend(missing)
        except Exception as e:
            self.logger.warning(f"[FineTune] Could not refresh optimizer param groups cleanly: {e}")

    def _log_param_overview(self, phase: str):
        total = sum(p.numel() for p in self._model.parameters())
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        self.logger.info(f"[FineTune] {phase}: trainable parameters {trainable:,}/{total:,}")
        self.logger.debug(
            "[FineTune] Body groups (early->late): " +
            " | ".join([f"g{i}({len(g)} tensors)" for i, g in enumerate(self._body_groups)])
        )
