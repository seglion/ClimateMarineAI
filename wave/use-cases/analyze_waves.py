"""
CAPA 2 — Use Case: Análisis de oleaje

Solo régimen medio + rosa de oleaje +  regimen extremal.
Toda la lógica de negocio vive aquí, NO en las entidades.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from entities.wave import (
    DirectionSector,
    MeanRegime,
    WaveRecord,
    WaveRose,
    WaveTimeSeries,
    ExtremeRegime)

from interfaces.repositories import WaveRepository, FigurePresenter


@dataclass
class WaveAnalysisResult:
    """Resultados."""
    series: WaveTimeSeries
    mean_regime: MeanRegime
    rose: WaveRose
    extreme_regime: ExtremeRegime
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
    ):
        self._repo = wave_repo
        self._figures = figure_presenter

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
        }

        return WaveAnalysisResult(
            series=series,
            mean_regime=mean_regime,
            rose=rose,
            figures=figures,
        )

    # ── Operaciones sobre series (lógica de negocio) ──

    @staticmethod
    def _valid_hs(series: WaveTimeSeries) -> np.ndarray:
        """Hs sin NaN, listo para operar."""
        hs = series.hs
        return hs[~np.isnan(hs)]

    @staticmethod
    def _percentile(hs: np.ndarray, p: float) -> float:
        return float(np.nanpercentile(hs, p))

    @staticmethod
    def _filter_above(
        series: WaveTimeSeries, threshold: float
    ) -> WaveTimeSeries:
        return WaveTimeSeries(
            records=tuple(
                r for r in series.records
                if not np.isnan(r.hs) and r.hs >= threshold
            ),
            source_id=series.source_id,
            longitude=series.longitude,
            latitude=series.latitude,
            source_type=series.source_type,
        )

    @staticmethod
    def _annual_maxima(series: WaveTimeSeries) -> list[WaveRecord]:
        by_year: dict[int, WaveRecord] = {}
        for r in series.records:
            if np.isnan(r.hs):
                continue
            year = r.timestamp.year
            if year not in by_year or r.hs > by_year[year].hs:
                by_year[year] = r
        return [by_year[y] for y in sorted(by_year)]

    # ── Cálculos estadísticos ──

    @staticmethod
    def _compute_mean_regime(
        series: WaveTimeSeries,
        percentiles: tuple[float, ...],
    ) -> MeanRegime:
        """Ajuste log-normal al régimen medio de Hs."""
        hs = series.hs[~np.isnan(series.hs)]
        hs = hs[hs > 0]

        log_hs = np.log(hs)
        mu = float(np.mean(log_hs))
        sigma = float(np.std(log_hs, ddof=1))

        dist = scipy_stats.lognorm(s=sigma, scale=np.exp(mu))
        computed = tuple(
            (p, float(dist.ppf(p / 100))) for p in percentiles
        )

        return MeanRegime(percentiles=computed, mu=mu, sigma=sigma)

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

        dist = []
        for sector in sectors:
            sector_mask = np.array([
                DirectionSector.from_degrees(d) == sector
                for d in dirs
            ]) & valid

            row = []
            hs_sector = hs[sector_mask]
            for i in range(len(hs_bins) - 1):
                lo, hi = hs_bins[i], hs_bins[i + 1]
                count = np.sum((hs_sector >= lo) & (hs_sector < hi))
                row.append(float(count / n_valid * 100) if n_valid > 0 else 0.0)
            dist.append(tuple(row))

        return WaveRose(
            sectors=tuple(sectors),
            hs_bins=hs_bins,
            distribution=tuple(dist),
        )
