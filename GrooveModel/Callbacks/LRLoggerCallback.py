from GrooveModel.Callbacks.Callback import Callback


class LRLoggerCallback(Callback):
    def __init__(self, logger=None):
        super().__init__(logger=logger)

    def _log_lr(self, learner):
        lrs = [group['lr'] for group in learner.optimizer.param_groups]
        lr_str = ", ".join(f"{lr:.6f}" for lr in lrs)
        self.logger.info(
            f"[LRLogger] Epoch {learner.epoch} (step {learner.global_step}) – Learning rate(s): {lr_str}"
        )

    def on_epoch_end(self, learner):
        self._log_lr(learner)
