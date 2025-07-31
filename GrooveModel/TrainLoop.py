import logging
from typing import Callable, Dict

import torch
from tqdm import tqdm

from GrooveModel.Callbacks import CallbackManager
from GrooveModel.LearnerState import LearnerState


def run_training_loop(
    learner: LearnerState,
    callback_manager: CallbackManager,
    device: torch.device,
    compute_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    compute_metrics_fn: Callable[[torch.Tensor, torch.Tensor], Dict[str, float]],
    logger: logging.Logger
) -> None:
    callback_manager.call("on_train_begin", learner)

    for epoch in range(learner.start_epoch, learner.max_epochs):
        learner.epoch = epoch
        callback_manager.call("on_epoch_begin", learner)

        # Training phase
        learner.model.train()
        total_loss = 0.0

        for batch in tqdm(learner.train_loader, desc=f"Epoch {epoch+1}"):
            callback_manager.call("on_batch_begin", learner)

            inputs, targets = batch[0].to(device), batch[1].to(device)
            learner.optimizer.zero_grad()

            outputs = learner.model(inputs)
            loss = compute_loss_fn(outputs, targets)
            loss.backward()
            learner.optimizer.step()

            total_loss += loss.item()
            callback_manager.call("on_batch_end", learner)

        learner.train_loss = total_loss / len(learner.train_loader)

        # Validation phase
        learner.model.eval()
        total_val_loss = 0.0
        all_metrics = {}

        with torch.no_grad():
            for batch in learner.val_loader:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = learner.model(inputs)
                loss = compute_loss_fn(outputs, targets)
                total_val_loss += loss.item()

                batch_metrics = compute_metrics_fn(outputs, targets)
                for k, v in batch_metrics.items():
                    all_metrics[k] = all_metrics.get(k, 0.0) + v

        learner.val_loss = total_val_loss / len(learner.val_loader)
        learner.metrics = {k: v / len(learner.val_loader) for k, v in all_metrics.items()}

        if learner.scheduler:
            learner.scheduler.step()

        callback_manager.call("on_epoch_end", learner)

        if callback_manager.state.get("early_stop", False):
            logger.warning("Early stopping triggered.")
            break

    callback_manager.call("on_train_end", learner)