import logging
from typing import Callable, Optional

import torch
from torch import optim
from torch.amp import autocast
from tqdm import tqdm

from GrooveModel.Callbacks.Callback import CallbackManager
from GrooveModel.Learner.LearnerState import LearnerState
from GrooveModel.Metrics import BaseMetrics


def run_training_loop(
        learner: LearnerState,
        callback_manager: CallbackManager,
        device: torch.device,
        compute_loss_fn: Callable[..., torch.Tensor],
        metrics: BaseMetrics,
        logger: Optional[logging.Logger] = None,
        use_mixed_precision: bool = False
) -> None:
    """Training loop with grad-clipping hooks, AMP, tqdm, and streaming metrics."""
    callback_manager.call("on_train_begin", learner)

    for epoch in range(learner.start_epoch, learner.max_epochs):
        learner.epoch = epoch
        callback_manager.call("on_epoch_begin", learner)

        # ---- Train ----
        learner.model.train()
        total_loss = 0.0

        for batch in tqdm(learner.train_loader, desc=f"Train Epoch {epoch}", leave=False):
            callback_manager.call("on_batch_begin", learner)

            inputs, targets, beat_pos = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            learner.optimizer.zero_grad(set_to_none=True)

            if use_mixed_precision:
                with autocast(device.type, dtype=torch.bfloat16):
                    outputs = learner.model((inputs, beat_pos))
                    loss = compute_loss_fn(outputs, targets)
            else:
                outputs = learner.model((inputs, beat_pos))
                loss = compute_loss_fn(outputs, targets)

            loss.backward()

            callback_manager.call("on_after_backward", learner)  # e.g., gradient clipping

            learner.optimizer.step()

            if learner.scheduler is not None and learner.step_based_scheduler:
                learner.scheduler.step()

            total_loss += float(loss.item())
            learner.global_step += 1
            callback_manager.call("on_batch_end", learner)

        learner.train_loss = total_loss / max(1, len(learner.train_loader))

        # ---- Validate ----
        learner.model.eval()
        total_val_loss = 0.0
        metrics.reset()

        with torch.no_grad():
            for batch in tqdm(learner.val_loader, desc=f"Validation Epoch {epoch}", leave=False):
                inputs, targets, beat_pos = batch[0].to(device), batch[1].to(device), batch[2].to(device)

                if use_mixed_precision:
                    with autocast(device.type, dtype=torch.bfloat16):
                        outputs = learner.model((inputs, beat_pos))
                        loss = compute_loss_fn(outputs, targets)
                else:
                    outputs = learner.model((inputs, beat_pos))
                    loss = compute_loss_fn(outputs, targets)

                total_val_loss += float(loss.item())
                metrics.update_batch(outputs, targets)

        num_val_batches = max(1, len(learner.val_loader))
        learner.val_loss = total_val_loss / num_val_batches
        learner.metrics = metrics.compute_all()

        # ---- Epoch schedulers ----
        if learner.scheduler is not None and not learner.step_based_scheduler:
            if isinstance(learner.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                learner.scheduler.step(learner.val_loss)
            else:
                learner.scheduler.step()

        callback_manager.call("on_epoch_end", learner)

        if callback_manager.state.get("early_stop", False):
            if logger:
                logger.warning("Early stopping triggered.")
            break

    callback_manager.call("on_train_end", learner)
