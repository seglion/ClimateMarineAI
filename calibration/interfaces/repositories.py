from __future__ import annotations

from abc import ABC, abstractmethod

from entities.calibration import CalibrationData, CalibrationResult


class InstrumentalRepository(ABC):
    """Carga pares (Hs_instrumental, Hs_reanalisis, dirección) alineados en tiempo.

    La responsabilidad de alinear temporalmente los datos instrumentales con
    los del reanálisis recae en el adaptador concreto, no en el caso de uso.
    """

    @abstractmethod
    def load(self, source_id: str) -> CalibrationData: ...


class CalibrationFitter(ABC):
    """Ajusta el modelo direccional Hs^C = α(θ)·(Hs^R)^β(θ).

    Encapsula el algoritmo de optimización (L-BFGS-B sobre cuantiles en escala
    de Gumbel) para que el caso de uso sea independiente de scipy.
    """

    @abstractmethod
    def fit(
        self,
        data: CalibrationData,
        n_nodes: int = 8,
        sector_width: float = 90.0,
    ) -> CalibrationResult: ...