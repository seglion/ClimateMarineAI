from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np

class DirectionSector(Enum):
    """Sectores direccionales de 22.5°."""
    N = 0.0;    NNE = 22.5;  NE = 45.0;   ENE = 67.5
    E = 90.0;   ESE = 112.5; SE = 135.0;  SSE = 157.5
    S = 180.0;  SSW = 202.5; SW = 225.0;  WSW = 247.5
    W = 270.0;  WNW = 292.5; NW = 315.0;  NNW = 337.5

    @classmethod
    def from_degrees(cls, deg: float) -> "DirectionSector":
        deg = deg % 360
        return min(
            cls,
            key=lambda s: min(abs(s.value - deg), 360 - abs(s.value - deg)),
        )

    @classmethod
    def all_sectors(cls) -> list["DirectionSector"]:
        return sorted(cls, key=lambda s: s.value)

@dataclass(frozen=True)
class WaveRecord:
    """Registro atómico de oleaje. Inmutable.
    Permite NaN para datos ausentes sin perder la marca temporal.
    """
    timestamp: datetime
    hs: float       # Altura significante (m). NaN = ausente.
    tp: float       # Periodo pico (s). NaN = ausente.
    dir: float      # Dirección procedencia (°). NaN = ausente.
    tm: float | None = None
    
@dataclass(frozen=True)
class WaveTimeSeries:
    """Serie temporal de oleaje. Puro contenedor.
    Puede contener NaN — el use case decide cómo manejarlos.
    """
    records: tuple[WaveRecord, ...]
    source_id: str
    longitude: float
    latitude: float
    source_type: str = "unknown"

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def start(self) -> datetime:
        return self.records[0].timestamp

    @property
    def end(self) -> datetime:
        return self.records[-1].timestamp

    @property
    def hs(self) -> np.ndarray:
        return np.array([r.hs for r in self.records])

    @property
    def tp(self) -> np.ndarray:
        return np.array([r.tp for r in self.records])

    @property
    def directions(self) -> np.ndarray:
        return np.array([r.dir for r in self.records])

    @property
    def timestamps(self) -> list[datetime]:
        return [r.timestamp for r in self.records]

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(~np.isnan(self.hs)))

    @property
    def n_gaps(self) -> int:
        return int(np.count_nonzero(np.isnan(self.hs)))

    @property
    def coverage(self) -> float:
        return self.n_valid / self.size * 100 if self.size > 0 else 0.0

@dataclass(frozen=True)
class MeanRegime:
    """Régimen medio escalar."""
    percentiles: tuple[tuple[float, float], ...]  # ((50, 1.04), (95, 2.66), ...)
    mu: float
    sigma: float
    mu_2: float
    sigma_2: float
    distribution: str = "lognormal"

    def hs_at(self, p: float) -> float:
        for perc, val in self.percentiles:
            if abs(perc - p) < 0.01:
                return val
        available = [perc for perc, _ in self.percentiles]
        raise KeyError(f"Percentil {p} no calculado. Disponibles: {available}")
    
@dataclass(frozen=True)
class ExtremeRegime:
    """Régimen Extremal escalar."""
    estadisticos: tuple[tuple[int, float,float,float], ...]  # ((Tr,Hs,Bi,Bs), (100, 2.66, 1.5, 3.0), ...)
    distribution: str = "maxima"
    pot:float | None = None

        
@dataclass(frozen=True)
class WaveRose:
    """Rosa de oleaje."""
    sectors: tuple[DirectionSector, ...]
    hs_bins: tuple[float, ...]
    distribution: tuple[tuple[float, ...], ...]  # [n_sectors × n_bins]
    threshold: float | None = None  # None = rosa completa, valor = percentil de corte
    @property
    def sector_frequencies(self) -> tuple[float, ...]:
        return tuple(sum(row) for row in self.distribution)

    @property
    def dominant_sector(self) -> DirectionSector:
        freqs = self.sector_frequencies
        return self.sectors[int(np.argmax(freqs))]