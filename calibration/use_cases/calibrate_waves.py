from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from entities.calibration import CalibrationData, CalibrationResult
from interfaces.calibration import CalibrationFitter, InstrumentalRepository


@dataclass
class CalibrateWavesResult:
    """Salida del caso de uso: modelo ajustado + serie corregida."""
    data: CalibrationData
    model: CalibrationResult
    hs_calibrated: np.ndarray


class CalibrateWaves:
    """Use case: calibración direccional de Hs de reanálisis (Mínguez et al., 2011).

    Paso 1 — carga pares (Hs^I, Hs^R, θ) ya alineados en tiempo.
    Paso 2 — ajusta el modelo α(θ)·(Hs^R)^β(θ) sobre cuantiles Gumbel.
    Paso 3 — aplica la corrección a la misma serie de reanálisis.
    """

    def __init__(
        self,
        instrumental_repo: InstrumentalRepository,
        fitter: CalibrationFitter,
    ):
        self._repo = instrumental_repo
        self._fitter = fitter

    def execute(
        self,
        source_id: str,
        n_nodes: int = 16,
        sector_width: float = 22.5,
    ) -> CalibrateWavesResult:
        data = self._repo.load(source_id)
        model = self._fitter.fit(data, n_nodes=n_nodes, sector_width=sector_width)
        hs_calibrated = self._apply(model, data.hs_reanalysis, data.directions)
        return CalibrateWavesResult(data=data, model=model, hs_calibrated=hs_calibrated)

    @staticmethod
    def _apply(
        model: CalibrationResult,
        hs_reanalysis: np.ndarray,
        directions: np.ndarray,
    ) -> np.ndarray:
        theta = directions % 360
        alpha = model.alpha_spline(theta)
        beta = model.beta_spline(theta)
        return alpha * hs_reanalysis ** beta
