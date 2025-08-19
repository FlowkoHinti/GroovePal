import json
import os
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List, Set

import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

from GrooveModel.Callbacks.Callback import Callback


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
