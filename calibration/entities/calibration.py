from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationData:
    """Pares (Hs_instrumental, Hs_reanalysis, dirección) alineados en tiempo.

    Hs^I — Mínguez et al. (2011) — datos de boya o satélite.
    Hs^R — misma marca temporal, del reanálisis a calibrar.
    θ    — dirección media del oleaje [°], normalizada a [0, 360).
    """
    hs_instrumental: np.ndarray
    hs_reanalysis: np.ndarray
    directions: np.ndarray

    def __post_init__(self) -> None:
        if not (len(self.hs_instrumental) == len(self.hs_reanalysis) == len(self.directions)):
            raise ValueError(
                "hs_instrumental, hs_reanalysis y directions deben tener la misma longitud."
            )
        object.__setattr__(self, "directions", self.directions % 360)


@dataclass(frozen=True)
class CalibrationDiagnostics:
    """Estadísticos diagnósticos antes y después de la calibración.

    Corresponden a las Ecs. (27)-(29) de Mínguez et al. (2011).
    """
    bias_reanalysis: float
    bias_calibrated: float
    rms_reanalysis: float
    rms_calibrated: float
    rsi_reanalysis: float
    rsi_calibrated: float
    r_reanalysis: float
    r_calibrated: float


@dataclass(frozen=True)
class ConfidenceIntervals:
    """Intervalos de confianza al nivel indicado para α y β en cada nodo.

    Ec. (22) de Mínguez et al. (2011).
    """
    alpha_lower: tuple[float, ...]
    alpha_upper: tuple[float, ...]
    beta_lower: tuple[float, ...]
    beta_upper: tuple[float, ...]
    level: float = 0.95


@dataclass
class CalibrationResult:
    """Resultado completo del ajuste direccional.

    Modelo: Hs^C = α(θ) · (Hs^R)^β(θ)   [Ec. 3 del paper]

    alpha_spline y beta_spline son CubicSpline periódicos de scipy;
    no son hashables, por eso la clase no es frozen.
    """
    alpha_nodes: np.ndarray       # α en cada nodo direccional
    beta_nodes: np.ndarray        # β en cada nodo direccional
    direction_nodes: np.ndarray   # θ_j [°] de los nodos

    alpha_spline: object          # CubicSpline: α(θ) continuo
    beta_spline: object           # CubicSpline: β(θ) continuo

    covariance_matrix: np.ndarray | None = None
    residual_variance: float | None = None
    confidence_intervals: ConfidenceIntervals | None = None
    diagnostics: CalibrationDiagnostics | None = None
