from GrooveModel.Callbacks.Callback import Callback


class LRLoggerCallback(Callback):
    def __init__(self, logger=None):
        super().__init__(logger=logger)

    def _log_lr(self, learner):
        lrs = []
        for i, group in enumerate(learner.optimizer.param_groups):
            name = group.get("name", f"group_{i}")  # default if no name was set
            lr = group["lr"]
            lrs.append(f"{name}={lr:.6f}")
        lr_str = ", ".join(lrs)
        self.logger.info(
            f"[LRLogger] Epoch {learner.epoch} (step {learner.global_step}) – Learning rate(s): {lr_str}"
        )

    def on_epoch_end(self, learner):
        self._log_lr(learner)
