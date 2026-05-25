from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from entities.wave import WaveRecord, WaveTimeSeries
from interfaces.repositories import WaveRepository

_NULL = -9999.9

_COL_DATE = "Fecha (GMT)"
_COL_HS   = "Altura Signif. del Oleaje(m)"
_COL_TM   = "Periodo Medio(s)"
_COL_TP   = "Periodo de Pico(s)"
_COL_DIR  = "Direcc. Media de Proced.(0=N,90=E)"


@dataclass(frozen=True)
class SimarSource:
    """Metadatos de una fuente SIMAR registrada."""
    source_id: str
    lon: float
    lat: float


class SimarRepository(WaveRepository):
    """Lee ficheros CSV de SIMAR (Puertos del Estado).

    Formato del fichero:
      - Línea 1: metadato  ("Valor nulo: -9999.9")
      - Línea 2: cabecera  (nombres de columnas, separador \\t)
      - Línea 3+: datos    (fecha "YYYY MM DD HH", valores numéricos)

    Las coordenadas (lon, lat) no están en el fichero — se registran
    explícitamente al construir el repositorio:

        repo = SimarRepository(data_dir=Path("data/"))
        repo.register(SimarSource("3020040", lon=-8.67, lat=43.67))
        repo.register(SimarSource("1052046", lon=-9.10, lat=44.00))

    El repositorio busca el fichero cuyo nombre contenga el source_id.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        self._sources: dict[str, SimarSource] = {}

    def register(self, source: SimarSource) -> None:
        self._sources[source.source_id] = source

    def load(
        self,
        source_id: str,
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
    ) -> WaveTimeSeries:
        if source_id not in self._sources:
            raise KeyError(
                f"source_id '{source_id}' no registrado. "
                f"Llama a repo.register(SimarSource(...)) primero."
            )
        meta = self._sources[source_id]
        path = self._find_file(source_id)
        df   = self._read_simar(path)

        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]

        hs_arr  = df["hs"].to_numpy(dtype=float)
        tp_arr  = df["tp"].to_numpy(dtype=float)
        dir_arr = df["dir"].to_numpy(dtype=float)
        tm_arr  = df["tm"].to_numpy(dtype=float)
        records = tuple(
            WaveRecord(
                timestamp=ts.to_pydatetime(),
                hs=float(hs),
                tp=float(tp),
                dir=float(dir_),
                tm=float(tm) if not np.isnan(tm) else None,
            )
            for ts, hs, tp, dir_, tm in zip(df.index, hs_arr, tp_arr, dir_arr, tm_arr)
        )

        return WaveTimeSeries(
            records=records,
            source_id=source_id,
            longitude=meta.lon,
            latitude=meta.lat,
            source_type="simar",
        )

    def list_sources(self) -> list[dict]:
        return [
            {
                "source_id": s.source_id,
                "lon": s.lon,
                "lat": s.lat,
                "file": self._find_file(s.source_id).name,
            }
            for s in self._sources.values()
        ]

    # ── helpers ──

    def _find_file(self, source_id: str) -> Path:
        matches = list(self._data_dir.glob(f"*{source_id}*.csv"))
        if not matches:
            raise FileNotFoundError(
                f"No se encontró fichero con id '{source_id}' en {self._data_dir}"
            )
        return matches[0]

    @staticmethod
    def _read_simar(path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path, sep="\t", skiprows=1, header=0)

        dates = pd.to_datetime(raw[_COL_DATE].str.strip(), format="%Y %m %d %H")

        df = pd.DataFrame(
            {
                "hs":  pd.to_numeric(raw[_COL_HS],  errors="coerce").values,
                "tp":  pd.to_numeric(raw[_COL_TP],  errors="coerce").values,
                "dir": pd.to_numeric(raw[_COL_DIR], errors="coerce").values,
                "tm":  pd.to_numeric(raw[_COL_TM],  errors="coerce").values,
            },
            index=dates,
        )

        df.replace(_NULL, np.nan, inplace=True)
        df[df < 0] = np.nan

        return df
