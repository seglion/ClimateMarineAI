from __future__ import annotations

import numpy as np
import pandas as pd
from pyextremes import EVA

from entities.wave import ExtremeRegime, WaveTimeSeries
from interfaces.repositories import ExtremeValueFitter


class PyextremesEVAFitter(ExtremeValueFitter):

    def fit_block_maxima(
        self,
        series: WaveTimeSeries,
        return_periods: tuple[int, ...] = (2, 5, 10, 25, 50, 100, 200, 500),
    ) -> ExtremeRegime:
        hs = self._to_series(series)
        model = EVA(hs)
        model.get_extremes(method="BM", block_size="365.2425D")
        model.fit_model()
        summary = model.get_summary(
            return_period=list(return_periods),
            alpha=0.95,
            n_samples=1000,
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
        hs = self._to_series(series)
        threshold = float(hs.quantile(threshold_percentile / 100))
        model = EVA(hs)
        model.get_extremes(method="POT", extremes_type="high", threshold=threshold)
        model.fit_model()
        summary = model.get_summary(
            return_period=list(return_periods),
            alpha=0.95,
            n_samples=1000,
        )
        estadisticos = tuple(
            (int(tr), float(row["return value"]), float(row["lower ci"]), float(row["upper ci"]))
            for tr, row in summary.iterrows()
        )
        return ExtremeRegime(estadisticos=estadisticos, distribution="POT", pot=threshold)

    @staticmethod
    def _to_series(series: WaveTimeSeries) -> pd.Series:
        return pd.Series(
            {r.timestamp: r.hs for r in series.records if not np.isnan(r.hs)}
        )