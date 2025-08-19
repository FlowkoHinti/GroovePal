import json
import logging
import os
import time
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Union, Optional, Literal, Dict, Any

import torch

from GrooveModel.Callbacks.Callback import Callback

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml

    _NVML_OK = True
except ImportError:
    _NVML_OK = False


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
        self.best_path = self.save_root / f"{model_name}_best.pt"
        self.stats_path = self.save_root / "training_stats.jsonl"

        self.state = {}
        self.state.setdefault("best_value", float("inf") if self.mode == "min" else -float("inf"))

        # per-epoch util samples
        self._gpu_util_sampled: Optional[float] = None
        self._cpu_util_sampled: Optional[float] = None

        # NVML (unchanged)
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

    # ---- Helpers (unchanged + new helper for loss vars) ----
    def _to_mb(self, bytes_val: Optional[int]) -> Optional[float]:
        return round(bytes_val / (1024 ** 2), 2) if bytes_val is not None else None

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

    def _extract_loss_log_vars(self, criterion: torch.nn.Module) -> Optional[Dict[str, float]]:
        """
        If the loss has learnable per-task log-variances (e.g., criterion.log_vars),
        record them for the stats JSONL (helps debugging/plotting).
        """
        try:
            if hasattr(criterion, "log_vars"):
                return {k: float(v.detach().cpu().item()) for k, v in criterion.log_vars.items()}
        except Exception:
            pass
        return None

    # ---- Hooks ----
    def on_train_begin(self, learner):
        self.logger.info(f"[Checkpoint] Saving to: {self.save_root}")

        best_ckpt = self._load_checkpoint(self.best_path)
        if best_ckpt is not None and "monitor_value" in best_ckpt:
            self.state["best_value"] = float(best_ckpt["monitor_value"])
            self.logger.info(f"[Checkpoint] Loaded best value={self.state['best_value']:.6f}")

        if self.load_best_on_start:
            ckpt = best_ckpt or self._load_checkpoint(self.latest_path)
            src = "best" if best_ckpt is not None else "latest"
        else:
            latest = self._load_checkpoint(self.latest_path)
            ckpt = latest or best_ckpt
            src = "latest" if latest is not None else "best"

        if ckpt is not None:
            self.logger.info(f"[Checkpoint] Resuming from {src} checkpoint (epoch {ckpt.get('epoch', -1)}).")
            learner.model.load_state_dict(ckpt["model_state_dict"])
            # NEW: restore criterion (loss) state if present
            if ckpt.get("criterion_state_dict") is not None:
                try:
                    learner.criterion.load_state_dict(ckpt["criterion_state_dict"])
                except Exception as e:
                    self.logger.warning(f"[Checkpoint] Criterion state not loaded: {e}")
            # Optimizer & scheduler (as before)
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

        # NEW: snapshot current loss weights if available
        loss_log_vars = self._extract_loss_log_vars(learner.criterion)

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
            "loss_log_vars": loss_log_vars,
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
            "criterion_state_dict": (
                learner.criterion.state_dict() if hasattr(learner.criterion, "state_dict") else None),
            "optimizer_state_dict": learner.optimizer.state_dict(),
            "scheduler_state_dict": (learner.scheduler.state_dict() if learner.scheduler else None),
        }

        # Save latest (atomic)
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
