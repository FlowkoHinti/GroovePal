from typing import Literal

from GrooveModel.Callbacks.Callback import Callback


class EarlyStoppingCallback(Callback):
    def __init__(
            self,
            monitor: str = "val_loss",
            patience: int = 5,
            min_delta: float = 0.0,
            mode: Literal["min", "max"] = "min",
            logger=None
    ):
        super().__init__(logger=logger)
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best = float("inf") if mode == "min" else -float("inf")
        self.wait = 0

    # --- helpers ---
    def _get_current(self, learner):
        # 1) attribute on LearnerState (e.g., val_loss)
        if hasattr(learner, self.monitor):
            return getattr(learner, self.monitor)
        # 2) metrics dict (e.g., metrics["accuracy"])
        if getattr(learner, "metrics", None) and self.monitor in learner.metrics:
            return learner.metrics[self.monitor]
        return None

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < (self.best - self.min_delta)
        else:
            return current > (self.best + self.min_delta)

    # --- callbacks ---
    def on_train_begin(self, learner):
        self.logger.info(
            f"[EarlyStopping] Monitoring '{self.monitor}' (mode='{self.mode}', "
            f"patience={self.patience}, min_delta={self.min_delta})."
        )

    def on_epoch_end(self, learner):
        current = self._get_current(learner)
        if current is None:
            self.logger.warning(
                f"[EarlyStopping] Monitor '{self.monitor}' not found in LearnerState or metrics; skipping."
            )
            return

        try:
            val = float(current)
        except Exception:
            self.wait += 1
            self.logger.info(f"[EarlyStopping] Non-numeric value; no improvement ({self.wait}/{self.patience}).")
            if self.wait >= self.patience:
                self.logger.warning(f"[EarlyStopping] Early stopping triggered at epoch {learner.epoch}.")
                self.state["early_stop"] = True
            return

        if val != val:  # NaN
            self.wait += 1
            self.logger.info(f"[EarlyStopping] Value is NaN; no improvement ({self.wait}/{self.patience}).")
        elif self._is_improvement(val):
            prev_best = self.best
            self.best = val
            self.wait = 0
            delta = (prev_best - val) if self.mode == "min" else (val - prev_best)
            self.logger.info(
                f"[EarlyStopping] Improvement: {self.monitor} {prev_best:.6f} -> {val:.6f} "
                f"(Δ={delta:.6f}) at epoch {learner.epoch}."
            )
        else:
            self.wait += 1
            self.logger.info(f"[EarlyStopping] No improvement ({self.wait}/{self.patience}).")

        if self.wait >= self.patience:
            self.logger.warning(f"[EarlyStopping] Early stopping triggered at epoch {learner.epoch}.")
            self.state["early_stop"] = True
