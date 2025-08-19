import json
from os import PathLike
from pathlib import Path
from typing import Union

from GrooveModel.Callbacks.Callback import Callback


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
