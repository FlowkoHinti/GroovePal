from typing import Optional, List

from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from GrooveModel.Callbacks.Callback import Callback


class GradientClippingCallback(Callback):
    """
    Clips gradients to avoid exploding gradients.
    - Apply after backward, before optimizer.step().
    - Logs a concise epoch summary instead of per-batch spam.
    """

    def __init__(
            self,
            max_norm: Optional[float] = None,
            max_value: Optional[float] = None,
            logger=None,
    ):
        super().__init__(logger=logger)
        self.max_norm = max_norm
        self.max_value = max_value

        if self.max_norm is None and self.max_value is None:
            raise ValueError("You must specify either max_norm or max_value.")

        # Per-epoch accumulators (not serialized)
        self._norms: List[float] = []
        self._clipped_count: int = 0
        self._batch_count: int = 0

    def on_after_backward(self, learner):
        """Run after loss.backward(), before optimizer.step()."""
        if self.max_norm is not None:
            total_norm = float(clip_grad_norm_(learner.model.parameters(), self.max_norm))
            self._norms.append(total_norm)
            if total_norm > self.max_norm:
                self._clipped_count += 1

        if self.max_value is not None:
            clip_grad_value_(learner.model.parameters(), self.max_value)

        self._batch_count += 1

    def on_epoch_end(self, learner):
        """Emit one summary line per epoch, then reset accumulators."""
        if self._batch_count == 0:
            return  # nothing to report

        parts = []
        if self.max_norm is not None and self._norms:
            avg_norm = sum(self._norms) / len(self._norms)
            max_norm_seen = max(self._norms)
            frac = self._clipped_count / self._batch_count
            parts.append(
                f"avg_norm={avg_norm:.4f}, max_norm_seen={max_norm_seen:.4f}, "
                f"clipped_batches={self._clipped_count}/{self._batch_count} ({frac:.0%}) "
                f"@ max_norm={self.max_norm}"
            )

        if self.max_value is not None:
            parts.append(f"per-element clamp @ max_value={self.max_value}")

        if parts:
            self.logger.debug(f"[GradClip] Epoch {learner.epoch} – " + " | ".join(parts))

        # reset for next epoch
        self._norms.clear()
        self._clipped_count = 0
        self._batch_count = 0
