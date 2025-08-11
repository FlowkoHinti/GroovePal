import json
import logging
import os
import time
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Union, Literal, Tuple, Dict, Optional, List, Any, Set

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from GrooveModel.LearnerState import LearnerState

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml

    _NVML_OK = True
except ImportError:
    _NVML_OK = False


class Callback:
    """Base class for all callbacks."""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("CallbackLogger")
        self.state = {}  # shared between cbs will be replaced with shared state via attach_state()


    def attach_state(self, shared_state: dict):
        """Attach the shared state dictionary from CallbackManager."""
        self.state = shared_state

    def on_train_begin(self, learner: LearnerState): pass

    def on_train_end(self, learner: LearnerState): pass

    def on_epoch_begin(self, learner: LearnerState): pass

    def on_epoch_end(self, learner: LearnerState): pass

    def on_batch_begin(self, learner: LearnerState): pass

    def on_batch_end(self, learner: LearnerState): pass

    def on_after_backward(self, learner: LearnerState): pass



class CallbackManager:
    """Manages a list of callbacks and shared state between them."""

    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.state = {}
        # Attach shared state once
        for cb in self.callbacks:
            if hasattr(cb, "attach_state"):
                cb.attach_state(self.state)
            else:
                cb.state = self.state  # fallback

    def call(self, method, *args, **kwargs):
        for cb in self.callbacks:
            getattr(cb, method)(*args, **kwargs)


class CheckpointCallback(Callback):
    def __init__(
        self,
        save_dir: Union[str, PathLike],
        model_name: str,
        device: Optional[torch.device] = None,
        logger: logging.Logger = None,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        load_best_on_start: bool = False,
    ):
        super().__init__(logger)
        self.save_root = Path(save_dir) / model_name / "checkpoints"
        self.save_root.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.device = device
        self.monitor = monitor
        self.mode = mode
        self.load_best_on_start = load_best_on_start

        self.latest_path = self.save_root / f"{model_name}_latest.pt"
        self.best_path   = self.save_root / f"{model_name}_best.pt"
        self.stats_path  = self.save_root / "training_stats.jsonl"

        self.state = {}
        self.state.setdefault("best_value", float("inf") if self.mode == "min" else -float("inf"))

        # per-epoch util samples
        self._gpu_util_sampled: Optional[float] = None
        self._cpu_util_sampled: Optional[float] = None

        # NVML
        self._nvml_init_done = False
        self._nvml_error = None
        if _NVML_OK:
            try:
                pynvml.nvmlInit()
                self._nvml_init_done = True
                drv = pynvml.nvmlSystemGetDriverVersion()
                self.logger.info(f"[Checkpoint] NVML initialized (driver {drv}).")
            except Exception as e:
                self._nvml_error = repr(e)
                self.logger.warning(f"[Checkpoint] NVML init failed: {self._nvml_error}")

    # ---- Helpers ----
    def _to_mb(self, bytes_val: Optional[int]) -> Optional[float]:
        return round(bytes_val / (1024**2), 2) if bytes_val is not None else None

    def _gpu_utilization_instant(self) -> Optional[float]:
        if not self._nvml_init_done:
            return None
        try:
            if self.device is None or self.device.type != "cuda":
                return None
            idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(idx))
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except Exception:
            return None

    def _peak_vram_bytes(self) -> Optional[int]:
        if self.device is None or not torch.cuda.is_available() or self.device.type != "cuda":
            return None
        try:
            return int(torch.cuda.max_memory_allocated(self.device))
        except Exception:
            return None

    def _reset_peak_vram(self):
        if self.device is None or not torch.cuda.is_available() or self.device.type != "cuda":
            return
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except Exception:
            pass

    def _ram_bytes(self) -> Optional[int]:
        if psutil is None:
            return None
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            return None

    def _cpu_util(self) -> Optional[float]:
        if psutil is None:
            return None
        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return None

    def _now_iso(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _monitored_value(self, learner) -> float:
        if self.monitor == "val_loss":
            return float(learner.val_loss)
        if self.monitor == "train_loss":
            return float(learner.train_loss)
        if self.monitor in learner.metrics:
            return float(learner.metrics[self.monitor])
        self.logger.warning(f"[Checkpoint] Monitor '{self.monitor}' not found; skipping best comparison.")
        return float("inf") if self.mode == "min" else -float("inf")

    def _is_better(self, candidate: float, best: float) -> bool:
        return candidate < best if self.mode == "min" else candidate > best

    def _save_jsonl(self, record: Dict[str, Any]):
        with open(self.stats_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_checkpoint(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e:
            self.logger.error(f"[Checkpoint] Failed to load '{path}': {e}")
            return None


    # ---- Hooks ----
    def on_train_begin(self, learner):
        self.logger.info(f"[Checkpoint] Saving to: {self.save_root}")

        best_ckpt = self._load_checkpoint(self.best_path)
        if best_ckpt is not None and "monitor_value" in best_ckpt:
            self.state["best_value"] = float(best_ckpt["monitor_value"])
            self.logger.info(f"[Checkpoint] Loaded best value={self.state['best_value']:.6f}")

        ckpt = None
        if self.load_best_on_start:
            ckpt = best_ckpt or self._load_checkpoint(self.latest_path)
            src = "best" if best_ckpt is not None else "latest"
        else:
            ckpt = self._load_checkpoint(self.latest_path) or best_ckpt
            src = "latest" if (ckpt is not None and ckpt is not best_ckpt) else "best"

        if ckpt is not None:
            self.logger.info(f"[Checkpoint] Resuming from {src} checkpoint (epoch {ckpt.get('epoch', -1)}).")
            learner.model.load_state_dict(ckpt["model_state_dict"])
            try:
                learner.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            except Exception as e:
                self.logger.warning(f"[Checkpoint] Optimizer state not loaded: {e}")
            if learner.scheduler is not None and ckpt.get("scheduler_state_dict"):
                try:
                    learner.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception as e:
                    self.logger.warning(f"[Checkpoint] Scheduler state not loaded: {e}")

            learner.global_step = int(ckpt.get("global_step", 0))
            last_epoch = int(ckpt.get("epoch", -1))
            learner.start_epoch = last_epoch + 1
            learner.epoch = learner.start_epoch

        self._reset_peak_vram()

    def on_epoch_begin(self, learner):
        self.state["epoch_start_time"] = time.time()
        self.state["epoch_start_iso"] = self._now_iso()
        self._gpu_util_sampled = None
        self._cpu_util_sampled = None
        self._reset_peak_vram()
        self.logger.info(f"[Checkpoint] Epoch {learner.epoch} started.")

    def on_batch_end(self, learner):
        if self._gpu_util_sampled is None:
            self._gpu_util_sampled = self._gpu_utilization_instant()
        if self._cpu_util_sampled is None:
            self._cpu_util_sampled = self._cpu_util()

    def on_epoch_end(self, learner):
        end_time = time.time()
        start_time = self.state.get("epoch_start_time", end_time)
        elapsed = end_time - start_time

        vram_peak_mb = self._to_mb(self._peak_vram_bytes())
        ram_rss_mb = self._to_mb(self._ram_bytes())
        gpu_util = self._gpu_util_sampled
        cpu_util = self._cpu_util_sampled

        stats_snapshot = {
            "epoch": int(learner.epoch),
            "global_step": int(learner.global_step),
            "start_time": self.state.get("epoch_start_iso", self._now_iso()),
            "end_time": self._now_iso(),
            "elapsed_sec": round(elapsed, 2),
            "train_loss": float(learner.train_loss),
            "val_loss": float(learner.val_loss),
            "metrics": dict(learner.metrics) if learner.metrics else {},
            "monitor": self.monitor,
            "mode": self.mode,
            "vram_peak_mb": vram_peak_mb,
            "ram_rss_mb": ram_rss_mb,
            "gpu_util_percent": gpu_util,
            "cpu_util_percent": cpu_util,
            "device": str(self.device) if self.device is not None else None,
        }

        monitor_value = self._monitored_value(learner)
        stats_snapshot["monitor_value"] = monitor_value

        ckpt = {
            "epoch": int(learner.epoch),
            "global_step": int(learner.global_step),
            "timestamp": self._now_iso(),
            "monitor": self.monitor,
            "monitor_value": monitor_value,
            "mode": self.mode,
            "metrics": dict(learner.metrics) if learner.metrics else {},
            "train_loss": float(learner.train_loss),
            "val_loss": float(learner.val_loss),
            "step_based_scheduler": bool(learner.step_based_scheduler),
            "model_state_dict": learner.model.state_dict(),
            "optimizer_state_dict": learner.optimizer.state_dict(),
            "scheduler_state_dict": (learner.scheduler.state_dict() if learner.scheduler else None),
        }

        # Save latest
        tmp_latest = self.latest_path.with_suffix(".pt.tmp")
        torch.save(ckpt, tmp_latest)
        os.replace(tmp_latest, self.latest_path)
        self.logger.info(
            f"[Checkpoint] Saved {self.model_name} latest (epoch {learner.epoch}, "
            f"{self.monitor}={monitor_value:.6f})."
        )

        # Save best if improved
        prev_best = self.state.get("best_value", float("inf") if self.mode == "min" else -float("inf"))
        if self._is_better(monitor_value, prev_best):
            self.state["best_value"] = monitor_value
            tmp_best = self.best_path.with_suffix(".pt.tmp")
            torch.save(ckpt, tmp_best)
            os.replace(tmp_best, self.best_path)
            self.logger.info(
                f"[Checkpoint] New BEST for {self.model_name} at epoch {learner.epoch}: "
                f"{prev_best:.6f} -> {monitor_value:.6f}"
            )

        # Append stats
        self._save_jsonl(stats_snapshot)

        parts = [
            f"elapsed={elapsed:.2f}s",
            f"VRAM_peak={vram_peak_mb}MB",
            f"RAM={ram_rss_mb}MB",
        ]
        if gpu_util is not None:
            parts.append(f"GPU~={gpu_util}%")
        if cpu_util is not None:
            parts.append(f"CPU~={cpu_util}%")

        self.logger.info("[Checkpoint] Stats: " + ", ".join(parts))

    def on_train_end(self, learner):
        self.logger.info(
            f"[Checkpoint] Training done. Latest: {self.latest_path.name}, "
            f"Best: {self.best_path.name if self.best_path.exists() else '—'}"
        )
        if self._nvml_init_done:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


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


# ---------- Plotting ----------
class PlotLossCurvesCallback(Callback):
    def __init__(
            self,
            save_dir: Union[str, os.PathLike],
            model_name: str,
            figsize: Tuple[int, int] = (10, 6),
            train_color: str = "blue",
            val_color: str = "orange",
            linewidth: float = 2.0,
            logger=None,
    ):
        super().__init__(logger=logger)
        self.plot_dir = Path(save_dir) / model_name / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.stats_file = Path(save_dir) / model_name / "checkpoints" / "training_stats.jsonl"
        self.figsize = figsize
        self.train_color = train_color
        self.val_color = val_color
        self.linewidth = linewidth

    # ---------- helpers ----------
    def _read_stats(self) -> List[dict]:
        if not self.stats_file.exists():
            self.logger.warning(f"[PlotLoss] Stats file not found: {self.stats_file}")
            return []
        stats: List[dict] = []
        with self.stats_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    stats.append(json.loads(line))
                except json.JSONDecodeError:
                    self.logger.warning("[PlotLoss] Skipping malformed JSONL line at end of file.")
        return stats

    def _extract_series(self, stats: List[dict]):
        epochs = [e.get("epoch") for e in stats]
        train_losses = [e.get("train_loss") for e in stats]
        val_losses = [e.get("val_loss") for e in stats]
        return epochs, train_losses, val_losses

    def _plot_losses(self, epochs: List[int], train: List[float], val: List[float]):
        plt.figure(figsize=self.figsize)
        plt.plot(epochs, train, label="Train Loss", marker="o",
                 color=self.train_color, linewidth=self.linewidth)
        plt.plot(epochs, val, label="Validation Loss", marker="o",
                 color=self.val_color, linewidth=self.linewidth)
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Loss Curves")
        plt.xticks(epochs)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    def _save_figure(self, name: str):
        png = self.plot_dir / f"{name}.png"
        svg = self.plot_dir / f"{name}.svg"
        plt.savefig(png, bbox_inches="tight")
        plt.savefig(svg, format="svg", bbox_inches="tight")
        plt.close()
        self.logger.info(f"[PlotLoss] Saved {name} to {png} and {svg}.")

    # ---------- callback ----------
    def on_epoch_end(self, learner):
        stats = self._read_stats()
        if not stats:
            self.logger.warning("[PlotLoss] No stats available to plot.")
            return

        epochs, train_losses, val_losses = self._extract_series(stats)
        if any(v is None for v in train_losses) or any(v is None for v in val_losses):
            self.logger.warning("[PlotLoss] Missing train/val loss values; plotting available points.")

        self._plot_losses(epochs, train_losses, val_losses)
        self._save_figure("loss_curves")


class PlotMetricsCallback(Callback):
    def __init__(
            self,
            save_dir: Union[str, os.PathLike],
            model_name: str,
            figsize: Tuple[int, int] = (10, 6),
            head_colors: Optional[Dict[str, str]] = None,
            line_width: float = 2.0,
            logger=None,
    ):
        super().__init__(logger=logger)
        self.plot_dir = Path(save_dir) / model_name / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.stats_file = Path(save_dir) / model_name / "checkpoints" / "training_stats.jsonl"
        self.figsize = figsize
        self.head_colors = head_colors or {}
        self.line_width = line_width

    # ---------- helpers: IO ----------
    def _read_stats(self) -> List[dict]:
        if not self.stats_file.exists():
            self.logger.warning(f"[PlotMetrics] Stats file not found: {self.stats_file}")
            return []
        stats: List[dict] = []
        with self.stats_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    stats.append(json.loads(line))
                except json.JSONDecodeError:
                    self.logger.warning("[PlotMetrics] Skipping malformed JSONL line at end of file.")
        return stats

    # ---------- helpers: processing ----------
    def _split_metrics(
            self, stats: List[dict]
    ) -> tuple[Dict[str, Set[str]], Set[str]]:
        """
        Returns:
          per_head: dict(metric -> set(heads)) for keys like 'head/metric'
          global_metrics: set(metric) for keys like 'metric'
        """
        per_head: Dict[str, Set[str]] = {}
        global_metrics: Set[str] = set()

        for entry in stats:
            m = entry.get("metrics") or {}
            for k in m.keys():
                if "/" in k:
                    head, metric = k.split("/", 1)
                    per_head.setdefault(metric, set()).add(head)
                else:
                    global_metrics.add(k)
        return per_head, global_metrics

    def _assign_colors_to_heads(self, heads: Set[str]):
        if self.head_colors:
            return
        base = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
        heads_sorted = sorted(heads)
        self.head_colors = {h: base[i % len(base)] for i, h in enumerate(heads_sorted)}

    # ---------- helpers: plotting ----------
    def _prepare_axes(self, title: str, ylabel: str, epochs: List[int], legend_title: Optional[str] = None):
        plt.figure(figsize=self.figsize)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(epochs)
        plt.grid(True)
        plt.tight_layout()
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(title=legend_title)

    def _save_metric_figure(self, metric: str, suffix: str = ""):
        base = f"{metric}_curve{suffix}"
        png = self.plot_dir / f"{base}.png"
        svg = self.plot_dir / f"{base}.svg"
        plt.savefig(png, format="png", bbox_inches="tight")
        plt.savefig(svg, format="svg", bbox_inches="tight")
        plt.close()
        self.logger.info(f"[PlotMetrics] Saved '{metric}{suffix}' to {png} and {svg}.")

    def _plot_series(
            self,
            epochs: List[int],
            values: List[Optional[float]],
            label: str,
            color: Optional[str] = None,
    ):
        if any(v is None for v in values):
            self.logger.warning(f"[PlotMetrics] Missing values for series '{label}'; skipping.")
            return False
        plt.plot(epochs, values, label=label, marker="o", linewidth=self.line_width, color=color)
        return True

    def _plot_per_head_metric(self, stats: List[dict], metric: str, heads: Set[str], epochs: List[int]):
        # collect colors
        self._assign_colors_to_heads(heads)
        # draw lines
        plotted_any = False
        for head in sorted(heads):
            key = f"{head}/{metric}"
            vals = [(e.get("metrics") or {}).get(key) for e in stats]
            plotted_any |= self._plot_series(epochs, vals, label=head, color=self.head_colors.get(head))

        if not plotted_any:
            self.logger.info(f"[PlotMetrics] No complete series for per-head metric '{metric}'; skipped.")
            return

        # labels & save
        plt.legend(title="Output Head")
        self._save_metric_figure(metric, suffix="")

    def _plot_global_metric(self, stats: List[dict], metric: str, epochs: List[int]):
        vals = [(e.get("metrics") or {}).get(metric) for e in stats]
        if not self._plot_series(epochs, vals, label=metric):
            return
        plt.legend()
        self._save_metric_figure(metric)

    # ---------- callback ----------
    def on_epoch_end(self, learner):
        stats = self._read_stats()
        if not stats:
            self.logger.warning("[PlotMetrics] No stats available to plot.")
            return

        epochs = [e.get("epoch") for e in stats]
        per_head, global_metrics = self._split_metrics(stats)

        # Per-head metrics
        if per_head:
            all_heads = set().union(*per_head.values())
            for metric, heads in per_head.items():
                self._prepare_axes(
                    title=f"{metric.title()} per Output Head",
                    ylabel=metric.title(),
                    epochs=epochs,
                    legend_title="Output Head",
                )
                self._plot_per_head_metric(stats, metric, heads, epochs)

        # Global (single-head) metrics
        for metric in sorted(global_metrics):
            self._prepare_axes(
                title=metric.title(),
                ylabel=metric.title(),
                epochs=epochs,
                legend_title=None,
            )
            self._plot_global_metric(stats, metric, epochs)


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


class EpochSummaryCallback(Callback):
    def __init__(
        self,
        save_dir: Union[str, PathLike],
        model_name: str,
        float_precision: int = 4,
        logger=None,
    ):
        super().__init__(logger=logger)
        self.stats_file = Path(save_dir) / model_name / "checkpoints" / "training_stats.jsonl"
        self.precision = int(float_precision)

    # ---- helpers ----
    def _read_last_record(self):
        if not self.stats_file.exists():
            self.logger.warning(f"[EpochSummary] Stats file not found: {self.stats_file}")
            return None
        last = None
        with self.stats_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    # tolerate a partially written last line
                    self.logger.warning("[EpochSummary] Skipping malformed JSONL line.")
                    continue
        if last is None:
            self.logger.warning("[EpochSummary] Stats file is empty.")
        return last

    def _fmt(self, x, digits=None, unit=""):
        if x is None:
            return "n/a"
        d = self.precision if digits is None else digits
        return f"{float(x):.{d}f}{unit}"

    # ---- hook ----
    def on_epoch_end(self, learner):
        rec = self._read_last_record()
        if rec is None:
            return

        # core parts
        parts = [
            f"Epoch {rec.get('epoch', learner.epoch)}",
            f"Train Loss: {self._fmt(rec.get('train_loss'))}",
            f"Val Loss: {self._fmt(rec.get('val_loss'))}",
        ]

        # average/global metrics (no slash)
        metrics = rec.get("metrics") or {}
        for name in sorted(k for k in metrics.keys() if "/" not in k):
            parts.append(f"{name}: {self._fmt(metrics[name])}")

        # timing + system
        parts.append(f"Elapsed: {self._fmt(rec.get('elapsed_sec'), digits=1, unit='s')}")
        parts.append(f"RAM: {self._fmt(rec.get('ram_rss_mb'), digits=1, unit='MB')}")
        parts.append(f"Peak VRAM: {self._fmt(rec.get('vram_peak_mb'), digits=1, unit='MB')}")

        # optional utils if present
        if rec.get("cpu_util_percent") is not None:
            parts.append(f"CPU: {self._fmt(rec.get('cpu_util_percent'), digits=0, unit='%')}")
        if rec.get("gpu_util_percent") is not None:
            parts.append(f"GPU: {self._fmt(rec.get('gpu_util_percent'), digits=0, unit='%')}")

        self.logger.info("[EpochSummary] " + " | ".join(parts))
