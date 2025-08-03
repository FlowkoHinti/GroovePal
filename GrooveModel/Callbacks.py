import json
import logging
import os
import time
from os import PathLike
from typing import Union, Literal, Tuple, Dict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import psutil
import torch
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from GrooveModel.LearnerState import LearnerState


class Callback:
    """Base class for all callbacks."""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("TrainerLogger")
        self.state = {}  # shared between all callbacks

    def on_train_begin(self, learner: LearnerState): pass

    def on_train_end(self, learner: LearnerState): pass

    def on_epoch_begin(self, learner: LearnerState): pass

    def on_epoch_end(self, learner: LearnerState): pass

    def on_batch_begin(self, learner: LearnerState): pass

    def on_batch_end(self, learner: LearnerState): pass


class CallbackManager:
    """Manages a list of callbacks and shared state between them."""

    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.state = {}

    def call(self, method, *args, **kwargs):
        for cb in self.callbacks:
            cb.state = self.state  # inject shared dict
            getattr(cb, method)(*args, **kwargs)


class CheckpointCallback(Callback):
    def __init__(
            self,
            save_dir: Union[str, PathLike],
            model_name: str,
            logger: logging.Logger = None,
            load_best: bool = False,
            monitor: str = "val_loss",
            mode: Literal["min", "max"] = "min"
    ):
        super().__init__(logger=logger)
        self.save_dir = os.path.join(save_dir, model_name, 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)

        self.stats_file = os.path.join(self.save_dir, 'training_stats.json')
        self.model_file = os.path.join(self.save_dir, 'model_state.pt')
        self.best_model_file = os.path.join(self.save_dir, 'best_model_state.pt')

        self.stats = []
        self.monitor = monitor
        self.mode = mode
        self.load_best = load_best
        self.best_metric_value = float("inf") if mode == "min" else -float("inf")
        self.last_epoch = 0
        self.epoch_start_time = None

    @staticmethod
    def _get_vram_usage():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated('cuda') / 1e6
            max_allocated = torch.cuda.max_memory_allocated('cuda') / 1e6
            return {'vram_MB': allocated, 'vram_max_MB': max_allocated}
        return {'vram_MB': 0.0, 'vram_max_MB': 0.0}

    def _is_better(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_metric_value
        else:
            return current > self.best_metric_value

    def on_train_begin(self, learner):
        if learner.start_epoch == 0 and os.path.exists(self.stats_file):
            self.logger.warning(f"Overwriting existing training stats at {self.stats_file}")

        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                self.stats = json.load(f)
                if self.stats:
                    last_record = self.stats[-1]
                    self.last_epoch = last_record['epoch']
                    file_to_load = self.best_model_file if self.load_best else self.model_file
                    self.logger.info(
                        f"Resuming from epoch {self.last_epoch + 1} using {'best' if self.load_best else 'latest'} model"
                    )
                    checkpoint = torch.load(file_to_load)
                    learner.model.load_state_dict(checkpoint['model'])
                    learner.optimizer.load_state_dict(checkpoint['optimizer'])
                    if learner.scheduler:
                        learner.scheduler.load_state_dict(checkpoint['scheduler'])
                    learner.start_epoch = self.last_epoch + 1
                    self.best_metric_value = last_record.get("best_metric_value", self.best_metric_value)

    def on_epoch_begin(self, learner):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, learner):
        elapsed = time.time() - self.epoch_start_time
        memory = psutil.Process(os.getpid()).memory_info().rss / 1e6  # RAM in MB
        vram = self._get_vram_usage()

        current_metric = getattr(learner, self.monitor, None)
        if current_metric is None:
            self.logger.warning(f"Monitored metric '{self.monitor}' not found in LearnerState.")
            current_metric = float("nan")
            is_best = False
        else:
            is_best = self._is_better(current_metric)

        record = {
            'epoch': learner.epoch,
            'train_loss': learner.train_loss,
            'val_loss': learner.val_loss,
            'metrics': learner.metrics,
            'elapsed_sec': elapsed,
            'memory_MB': memory,
            **vram,
            'monitored_value': current_metric,
            'best_metric_name': self.monitor,
            'best_metric_value': self.best_metric_value,
            'is_best': is_best,
            'model_path': os.path.relpath(self.model_file, start=self.save_dir),
            'best_model_path': os.path.relpath(self.best_model_file, start=self.save_dir)
        }

        self.stats.append(record)

        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

        # Always save latest
        torch.save({
            'model': learner.model.state_dict(),
            'optimizer': learner.optimizer.state_dict(),
            'scheduler': learner.scheduler.state_dict() if learner.scheduler else {},
        }, self.model_file)

        # Save best if improved
        if is_best:
            self.logger.info(
                f"New best model at epoch {learner.epoch} with {self.monitor} = {current_metric:.4f}"
            )
            self.best_metric_value = current_metric
            torch.save({
                'model': learner.model.state_dict(),
                'optimizer': learner.optimizer.state_dict(),
                'scheduler': learner.scheduler.state_dict() if learner.scheduler else {},
            }, self.best_model_file)
        else:
            self.logger.debug(
                f"Epoch {learner.epoch} completed. Train: {learner.train_loss:.4f}, Val: {learner.val_loss:.4f}"
            )


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

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < (self.best - self.min_delta)
        else:
            return current > (self.best + self.min_delta)

    def on_epoch_end(self, learner):
        current = getattr(learner, self.monitor, None)
        if current is None:
            self.logger.warning(f"Metric '{self.monitor}' not found in LearnerState.")
            return

        if self._is_improvement(current):
            self.best = current
            self.wait = 0
            self.logger.debug(f"EarlyStopping: Improvement detected: {self.monitor} = {current:.4f}")
        else:
            self.wait += 1
            self.logger.debug(f"EarlyStopping: No improvement ({self.wait}/{self.patience})")

            if self.wait >= self.patience:
                self.logger.warning(f"Early stopping triggered at epoch {learner.epoch}.")
                self.state["early_stop"] = True


# ---------- Plotting ----------
class PlotLossCurvesCallback(Callback):
    def __init__(
            self,
            save_dir: Union[str, PathLike],
            model_name: str,
            figsize: Tuple[int, int] = (8, 6),
            train_color: str = "blue",
            val_color: str = "orange",
            ymin: float = 0.0,
            ymax: float = 2.0,
            linewidth: float = 2.0,
            logger=None
    ):
        super().__init__(logger=logger)
        self.plot_dir = os.path.join(save_dir, model_name, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)

        self.checkpoint_dir = os.path.join(save_dir, model_name, 'checkpoints')
        self.stats_file = os.path.join(self.checkpoint_dir, 'training_stats.json')

        self.figsize = figsize
        self.train_color = train_color
        self.val_color = val_color
        self.ymin = ymin
        self.ymax = ymax
        self.linewidth = linewidth

    def on_epoch_end(self, learner):
        if not os.path.exists(self.stats_file):
            self.logger.warning(f"Stats file not found: {self.stats_file}")
            return

        with open(self.stats_file, 'r') as f:
            stats = json.load(f)

        epochs = [entry['epoch'] for entry in stats]
        train_losses = [entry['train_loss'] for entry in stats]
        val_losses = [entry['val_loss'] for entry in stats]

        plt.figure(figsize=self.figsize)
        plt.plot(epochs, train_losses, label="Train Loss", marker='o', color=self.train_color, linewidth=self.linewidth)
        plt.plot(epochs, val_losses, label="Validation Loss", marker='o', color=self.val_color, linewidth=self.linewidth)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plt.xticks(epochs)
        plt.ylim(self.ymin, self.ymax)
        plt.legend()
        plt.grid(True)

        png_path = os.path.join(self.plot_dir, "loss_curves.png")
        svg_path = os.path.join(self.plot_dir, "loss_curves.svg")

        plt.savefig(png_path, bbox_inches="tight")
        plt.savefig(svg_path, format="svg", bbox_inches="tight")
        plt.close()

        self.logger.debug(f"Loss curve updated and saved to {png_path} & {svg_path}")


class PlotMetricsCallback(Callback):
    def __init__(
        self,
        save_dir: Union[str, os.PathLike],
        model_name: str,
        figsize: Tuple[int, int] = (8, 6),
        metric_colormaps: Dict[str, str] = None,  # e.g., {"accuracy": "Blues", "f1": "Purples"}
        ymin: float = 0.0,
        ymax: float = 1.0,
        line_width: float = 2.0,
        logger=None
    ):
        super().__init__(logger=logger)
        self.plot_dir = os.path.join(save_dir, model_name, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)

        self.checkpoint_dir = os.path.join(save_dir, model_name, 'checkpoints')
        self.stats_file = os.path.join(self.checkpoint_dir, 'training_stats.json')

        self.figsize = figsize
        self.metric_colormaps = metric_colormaps or {
            "accuracy": "Blues",
            "precision": "Greens",
            "recall": "Oranges",
            "f1": "Purples",
            "perplexity": "Reds"
        }
        self.ymin = ymin
        self.ymax = ymax
        self.line_width = line_width

    def on_epoch_end(self, learner):
        if not os.path.exists(self.stats_file):
            self.logger.warning(f"Stats file not found: {self.stats_file}")
            return

        with open(self.stats_file, 'r') as f:
            stats = json.load(f)

        if not stats or not stats[0].get("metrics"):
            self.logger.warning("No metrics found in training stats.")
            return

        # Extract per-head metrics (those with slash)
        per_head_metrics = {}
        for key in stats[0]["metrics"].keys():
            if '/' in key:
                head, metric = key.split('/', 1)
                per_head_metrics.setdefault(metric, []).append(head)

        for metric, heads in per_head_metrics.items():
            plt.figure(figsize=self.figsize)
            epochs = [entry['epoch'] for entry in stats]

            cmap_name = self.metric_colormaps.get(metric, 'Blues')
            cmap = cm.get_cmap(cmap_name, len(heads))

            for i, head in enumerate(sorted(heads)):
                key = f"{head}/{metric}"
                values = [entry["metrics"].get(key) for entry in stats]

                if any(v is None for v in values):
                    self.logger.warning(f"Missing values for metric '{key}'")
                    continue

                plt.plot(
                    epochs,
                    values,
                    label=head,
                    color=cmap(i),
                    linewidth=self.line_width,
                    marker='o'
                )

            plt.xlabel("Epoch")
            plt.ylabel(metric.title())
            plt.title(f"{metric.title()} per Head")
            plt.xticks(epochs)
            plt.ylim(self.ymin, self.ymax)
            plt.legend(title="Head")
            plt.grid(True)

            png_path = os.path.join(self.plot_dir, f"{metric}_curve.png")
            svg_path = os.path.join(self.plot_dir, f"{metric}_curve.svg")

            plt.savefig(png_path, format="png", bbox_inches="tight")
            plt.savefig(svg_path, format="svg", bbox_inches="tight")
            plt.close()

            self.logger.debug(f"{metric.title()} plot saved to {png_path} & {svg_path}")


class LRLoggerCallback(Callback):
    def __init__(self, logger=None, log_every: str = "epoch"):  # or "batch"
        super().__init__(logger=logger)
        self.log_every = log_every

    def _log_lr(self, learner):
        lrs = [group['lr'] for group in learner.optimizer.param_groups]
        lr_str = ", ".join([f"{lr:.6f}" for lr in lrs])
        self.logger.info(f"[Epoch {learner.epoch}] Learning rate(s): {lr_str}")

    def on_epoch_end(self, learner):
        if self.log_every == "epoch":
            self._log_lr(learner)

    def on_batch_end(self, learner):
        if self.log_every == "batch":
            self._log_lr(learner)


class GradientClippingCallback(Callback):
    def __init__(self, max_norm: float = None, max_value: float = None, logger=None):
        super().__init__(logger=logger)
        self.max_norm = max_norm
        self.max_value = max_value

        if self.max_norm is None and self.max_value is None:
            raise ValueError("You must specify either max_norm or max_value.")

    def on_batch_end(self, learner):
        if self.max_norm is not None:
            clip_grad_norm_(learner.model.parameters(), self.max_norm)
        if self.max_value is not None:
            clip_grad_value_(learner.model.parameters(), self.max_value)


class EpochSummaryCallback(Callback):
    def __init__(
            self,
            save_dir: Union[str, PathLike],
            model_name: str,
            float_precision: int = 4,
            logger=None
    ):
        super().__init__(logger=logger)
        self.stats_file = os.path.join(save_dir, model_name, 'checkpoints', 'training_stats.json')
        self.precision = float_precision

    def on_epoch_end(self, learner):
        if not os.path.exists(self.stats_file):
            self.logger.warning(f"Stats file not found: {self.stats_file}")
            return

        with open(self.stats_file, 'r') as f:
            stats = json.load(f)

        if not stats:
            self.logger.warning("Stats file is empty.")
            return

        record = stats[-1]  # most recent epoch

        parts = [
            f"Epoch {record['epoch']}",
            f"Train Loss: {record['train_loss']:.{self.precision}f}",
            f"Val Loss: {record['val_loss']:.{self.precision}f}"
        ]

        # Filter to only include average metrics (no slash)
        metrics = record.get("metrics", {})
        avg_metrics = {k: v for k, v in metrics.items() if '/' not in k}

        for name, value in avg_metrics.items():
            parts.append(f"{name}: {value:.{self.precision}f}")

        parts.append(f"Elapsed: {record['elapsed_sec']:.1f}s")
        parts.append(f"RAM: {record['memory_MB']:.1f}MB")
        parts.append(f"VRAM: {record.get('vram_MB', 0.0):.1f}MB (max: {record.get('vram_max_MB', 0.0):.1f}MB)")

        self.logger.info(" | ".join(parts))
