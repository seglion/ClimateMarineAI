from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from entities.wave import WaveTimeSeries, MeanRegime, WaveRose, ExtremeRegime


class WaveRepository(ABC):

    @abstractmethod
    def load(
        self,
        source_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> WaveTimeSeries: ...

    @abstractmethod
    def list_sources(self) -> list[dict]: ...


class FigurePresenter(ABC):

    @abstractmethod
    def wave_rose(self, rose: WaveRose, title: str, path: Path) -> Path: ...

    @abstractmethod
    def extreme_wave_rose(self, rose: WaveRose, title: str, path: Path) -> Path: ...


    @abstractmethod
    def mean_regime(
        self, regime: MeanRegime, hs_data: ..., path: Path
    ) -> Path: ...
    
    @abstractmethod
    def extreme_regime(
        self, regime: ExtremeRegime, hs_data: ..., path: Path
    ) -> Path: ...