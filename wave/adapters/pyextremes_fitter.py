from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyextremes import EVA

from entities.wave import ExtremeRegime, WaveTimeSeries
from interfaces.repositories import ExtremeValueFitter

_PLOT_PERIODS = np.logspace(0.01, 2.6989701, 100)
_BLOCK_SIZE   = "365.2425D"
_COLOR_BM     = "#5157ff"
_COLOR_POT    = "#ff5151"


class PyextremesEVAFitter(ExtremeValueFitter):

    def __init__(self) -> None:
        self._cache_bm:  dict[str, EVA] = {}
        self._cache_pot: dict[str, EVA] = {}

    def fit_block_maxima(
        self,
        series: WaveTimeSeries,
        return_periods: tuple[int, ...] = (2, 5, 10, 25, 50, 100, 200, 500),
    ) -> ExtremeRegime:
        model = self._get_bm_model(series)
        summary = model.get_summary(
            return_period=list(return_periods),
            alpha=0.95,
            n_samples=500,
        )
        estadisticos = tuple(
            (int(tr), float(row["return value"]), float(row["lower ci"]), float(row["upper ci"]))
            for tr, row in summary.iterrows()
        )
        return ExtremeRegime(estadisticos=estadisticos, distribution="maxima")

    def fit_pot(
        self,
        series: WaveTimeSeries,
        threshold_percentile: float = 99.5,
        return_periods: tuple[int, ...] = (2, 5, 10, 25, 50, 100, 200, 500),
    ) -> ExtremeRegime:
        model, threshold = self._get_pot_model(series, threshold_percentile)
        summary = model.get_summary(
            return_period=list(return_periods),
            alpha=0.95,
            n_samples=500,
        )
        estadisticos = tuple(
            (int(tr), float(row["return value"]), float(row["lower ci"]), float(row["upper ci"]))
            for tr, row in summary.iterrows()
        )
        return ExtremeRegime(estadisticos=estadisticos, distribution="POT", pot=threshold)

    def plot_return_values_bm(self, series: WaveTimeSeries, path: Path) -> Path:
        model = self._get_bm_model(series)
        fig, ax = model.plot_return_values(
            return_period=_PLOT_PERIODS,
            return_period_size=_BLOCK_SIZE,
            alpha=0.95,
        )
        ax.set_xlabel("Periodo de retorno (años)")
        for i in range(min(3, len(ax.lines))):
            ax.lines[i].set_color(_COLOR_BM)
        if len(ax.collections) > 1:
            ax.collections[1].set_facecolor(_COLOR_BM)
        ax.set_title(f"Régimen Extremal BM (GEV) — {series.source_id}")
        ax.grid(True)
        ax.set_ylim(0, 16)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_return_values_pot(
        self,
        series: WaveTimeSeries,
        path: Path,
        threshold_percentile: float = 99.5,
    ) -> Path:
        model, threshold = self._get_pot_model(series, threshold_percentile)
        fig, ax = model.plot_return_values(
            return_period=_PLOT_PERIODS,
            return_period_size=_BLOCK_SIZE,
            alpha=0.95,
        )
        ax.set_xlabel("Periodo de retorno (años)")
        for i in range(min(3, len(ax.lines))):
            ax.lines[i].set_color(_COLOR_POT)
        if len(ax.collections) > 1:
            ax.collections[1].set_facecolor(_COLOR_POT)
        ax.set_title(
            f"Régimen Extremal POT (GPD) — {series.source_id} "
            f"— umbral = {threshold:.2f} m ({threshold_percentile}%)"
        )
        ax.grid(True)
        ax.set_ylim(0, 16)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_bm_model(self, series: WaveTimeSeries) -> EVA:
        sid = series.source_id
        if sid not in self._cache_bm:
            hs = self._to_series(series)
            model = EVA(hs)
            model.get_extremes(method="BM", block_size=_BLOCK_SIZE)
            model.fit_model()
            self._cache_bm[sid] = model
        return self._cache_bm[sid]

    def _get_pot_model(
        self, series: WaveTimeSeries, threshold_percentile: float = 99.5
    ) -> tuple[EVA, float]:
        sid = series.source_id
        if sid not in self._cache_pot:
            hs = self._to_series(series)
            threshold = float(hs.quantile(threshold_percentile / 100))
            model = EVA(hs)
            model.get_extremes(method="POT", extremes_type="high", threshold=threshold)
            model.fit_model()
            self._cache_pot[sid] = (model, threshold)
        return self._cache_pot[sid]

    @staticmethod
    def _to_series(series: WaveTimeSeries) -> pd.Series:
        hs = series.hs
        idx = pd.DatetimeIndex(series.timestamps)
        return pd.Series(hs, index=idx, dtype=float).dropna()
