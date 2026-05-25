"""
CAPA 2 — Use Case: Análisis de oleaje

Solo régimen medio + rosa de oleaje +  regimen extremal.
Toda la lógica de negocio vive aquí, NO en las entidades.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from entities.wave import (
    DirectionSector,
    MeanRegime,
    WaveRose,
    WaveTimeSeries,
    ExtremeRegime)

from interfaces.repositories import WaveRepository, FigurePresenter, ExtremeValueFitter


@dataclass
class WaveAnalysisResult:
    """Resultados."""
    series: WaveTimeSeries
    mean_regime: MeanRegime
    extreme_regime_bm: ExtremeRegime
    extreme_regime_pot: ExtremeRegime
    rose: WaveRose
    extreme_rose: WaveRose
    figures: dict[str, Path]


class AnalyzeWaves:
    """Use case: análisis de oleaje.

    Recibe dependencias por constructor (inyección).
    No sabe si el repo lee CSV, Parquet o una API.
    No sabe si el presenter usa matplotlib o plotly.
    """

    def __init__(
        self,
        wave_repo: WaveRepository,
        figure_presenter: FigurePresenter,
        extreme_fitter: ExtremeValueFitter,
    ):
        self._repo = wave_repo
        self._figures = figure_presenter
        self._eva = extreme_fitter

    def execute(
        self,
        source_id: str,
        output_dir: Path,
        percentiles: tuple[float, ...] = (50, 80, 85, 90, 95, 99, 99.5, 99.9),
    ) -> WaveAnalysisResult:

        # 1. Obtener datos
        series = self._repo.load(source_id)

        # 2. Calcular
        mean_regime = self._compute_mean_regime(series, percentiles)
        rose = self._compute_wave_rose(series)
        extreme_rose = self._compute_extreme_wave_rose(series, threshold=mean_regime.percentiles[-4][1])
        extreme_regime_bm = self._eva.fit_block_maxima(series, return_periods=(2, 5, 10, 25, 50, 100, 200, 500))
        extreme_regime_pot = self._eva.fit_pot(series, threshold_percentile=99.5, return_periods=(2, 5, 10, 25, 50, 100, 200, 500))

        # 3. Generar figuras
        output_dir.mkdir(parents=True, exist_ok=True)
        figures = {
            "wave_rose": self._figures.wave_rose(
                rose,
                f"Rosa de oleaje — {series.source_id}",
                output_dir / "rosa_oleaje.png",
            ),
            "mean_regime": self._figures.mean_regime(
                mean_regime,
                self._valid_hs(series),
                output_dir / "regimen_medio.png",
            ),
            "extreme_wave_rose": self._figures.extreme_wave_rose(
                extreme_rose,
                f"Rosa de oleaje extremo — {series.source_id}",
                output_dir / "rosa_oleaje_extremo.png",
            ),
            "extreme_regime_bm": self._eva.plot_return_values_bm(
                series,
                output_dir / "regimen_extremal_bm.png",
            ),
            "extreme_regime_pot": self._eva.plot_return_values_pot(
                series,
                output_dir / "regimen_extremal_pot.png",
            ),
        }

        return WaveAnalysisResult(
            series=series,
            mean_regime=mean_regime,
            extreme_regime_bm=extreme_regime_bm,
            extreme_regime_pot=extreme_regime_pot,
            rose=rose,
            extreme_rose=extreme_rose,
            figures=figures,
        )

    # ── Operaciones sobre series (lógica de negocio) ──

    @staticmethod
    def _valid_hs(series: WaveTimeSeries) -> np.ndarray:
        """Hs sin NaN, listo para operar."""
        hs = series.hs
        return hs[~np.isnan(hs)]

    @staticmethod
    def _filter_above(
        series: WaveTimeSeries, threshold: float
    ) -> WaveTimeSeries:
        indices = np.where((series.hs >= threshold) & ~np.isnan(series.hs))[0]
        return WaveTimeSeries(
            records=tuple(series.records[i] for i in indices),
            source_id=series.source_id,
            longitude=series.longitude,
            latitude=series.latitude,
            source_type=series.source_type,
        )

    # ── Cálculos estadísticos ──

    @staticmethod
    def _compute_mean_regime(
        series: WaveTimeSeries,
        percentiles: tuple[float, ...],
        discretizacion: float = 0.02,
        split_percentile: float = 80.0,
    ) -> MeanRegime:
        hs = series.hs[~np.isnan(series.hs)]
        hs = hs[hs > 0]

        bins = np.arange(hs.min(), hs.max(), discretizacion)
        N = np.histogram(hs, bins=bins)[0]
        centers = (bins[:-1] + bins[1:]) / 2
        bin_width = float(centers[1] - centers[0])

        area = float(np.sum(bin_width * N))
        n = N / area
        P1 = np.cumsum(bin_width * n)
        P11 = P1[:-1]
        y1 = centers[:-1]

        computed = tuple(
            (p, float(np.interp(p / 100, P11, y1))) for p in percentiles
        )

        # población 1: datos por debajo del percentil de corte (mar ordinario)
        split = float(np.nanpercentile(hs, split_percentile))
        hs_low = hs[hs <= split]
        log_low = np.log(hs_low)
        mu = float(np.mean(log_low))
        sigma = float(np.std(log_low, ddof=1))

        # población 2: cola superior (temporal / swell extremo)
        hs_high = hs[hs > split]
        log_high = np.log(hs_high)
        mu_2 = float(np.mean(log_high))
        sigma_2 = float(np.std(log_high, ddof=1))

        return MeanRegime(
            percentiles=computed,
            mu=mu, sigma=sigma,
            mu_2=mu_2, sigma_2=sigma_2,
        )

    @staticmethod
    def _compute_wave_rose(
        series: WaveTimeSeries,
        hs_bins: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 99.0),
    ) -> WaveRose:
        """Rosa de oleaje por sectores de 22.5°."""
        sectors = DirectionSector.all_sectors()
        hs = series.hs
        dirs = series.directions
        valid = ~np.isnan(hs) & ~np.isnan(dirs)
        n_valid = int(np.sum(valid))

        hs_v   = hs[valid]
        dirs_v = dirs[valid]

        # Asigna cada dirección a un índice de sector (0-15) vectorizado,
        # evitando 16×n llamadas Python a from_degrees().
        sector_idx = np.round(dirs_v / 22.5).astype(int) % 16

        bins_arr = np.array(hs_bins)
        dist = []
        for i in range(len(sectors)):
            counts = np.histogram(hs_v[sector_idx == i], bins=bins_arr)[0]
            row = tuple(float(c / n_valid * 100) if n_valid > 0 else 0.0 for c in counts)
            dist.append(row)

        return WaveRose(
            sectors=tuple(sectors),
            hs_bins=hs_bins,
            distribution=tuple(dist),
        )

    @staticmethod
    def _compute_extreme_wave_rose(
        series: WaveTimeSeries,
        threshold: float,
        step: float = 1.0,
        max_hs: float = 12.0,
    ) -> WaveRose:
        filtered = AnalyzeWaves._filter_above(series, threshold)
        t = float(np.round(threshold))
        hs_bins = tuple(float(x) for x in np.arange(t, max_hs, step)) + (99.0,)
        return AnalyzeWaves._compute_wave_rose(filtered, hs_bins)

