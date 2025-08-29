import json
import os
from pathlib import Path
from typing import Union, Tuple, List

from matplotlib import pyplot as plt

from GrooveModel.Callbacks.Callback import Callback


class PlotLossCurvesCallback(Callback):
    def __init__(
            self,
            save_dir: Union[str, os.PathLike],
            model_name: str,
            loss_label: str,
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
        self.loss_label = loss_label

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
        plt.plot(epochs, train, label="Train Objective", marker="o",
                 color=self.train_color, linewidth=self.linewidth)
        plt.plot(epochs, val, label="Val Objective", marker="o",
                 color=self.val_color, linewidth=self.linewidth)
        plt.xlabel("Epoch")
        plt.ylabel(self.loss_label)
        plt.title("Objective/Loss Curves")
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
