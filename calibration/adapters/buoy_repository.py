from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from entities.calibration import CalibrationData
from entities.wave import WaveTimeSeries
from interfaces.calibration import InstrumentalRepository

_NULL = -9999.9

_COL_DATE = "Fecha (GMT)"
_COL_HS   = "Altura Signif. del Oleaje(m)"
_COL_TP   = "Periodo de Pico(s)"


class PuertosEstadoBuoyRepository(InstrumentalRepository):
    """Lee ficheros de boya de Puertos del Estado y los alinea con el reanálisis.

    Formato del fichero (idéntico al SIMAR):
      - Línea 1: "Valor nulo: -9999.9"
      - Línea 2: cabecera \\t-separada
      - Línea 3+: datos con fecha "YYYY MM DD HH"

    La dirección se toma del reanálisis porque la mayoría de boyas no la
    miden o la tienen con muchos nulos.

    Args:
        reanalysis:        Serie de reanálisis ya cargada (SIMAR, GOW, ERA5…).
        data_dir:          Directorio donde buscar los CSV de boya.
        tolerance_minutes: Máxima diferencia temporal para considerar par válido.
    """

    def __init__(
        self,
        reanalysis: WaveTimeSeries,
        data_dir: Path,
        tolerance_minutes: int = 30,
    ) -> None:
        self._rean = reanalysis
        self._data_dir = Path(data_dir)
        self._tol = pd.Timedelta(minutes=tolerance_minutes)

    def load(self, source_id: str) -> CalibrationData:
        """Carga el CSV de boya e intersecta con el reanálisis.

        Args:
            source_id: Nombre o id del fichero de boya (sin extensión o completo).
                       El repositorio busca '*{source_id}*.csv' en data_dir.
        """
        path = self._find_file(source_id)
        df_inst = self._read_buoy(path)
        df_rean = self._reanalysis_frame()

        merged = pd.merge_asof(
            df_inst.sort_index(),
            df_rean.sort_index(),
            left_index=True,
            right_index=True,
            tolerance=self._tol,
            direction="nearest",
        ).dropna()

        if merged.empty:
            raise ValueError(
                f"No hay pares válidos entre '{source_id}' y el reanálisis "
                f"'{self._rean.source_id}' con tolerancia {self._tol}."
            )

        return CalibrationData(
            hs_instrumental=merged["hs_inst"].to_numpy(dtype=float),
            hs_reanalysis=merged["hs_rean"].to_numpy(dtype=float),
            directions=merged["dir_rean"].to_numpy(dtype=float),
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _find_file(self, source_id: str) -> Path:
        matches = list(self._data_dir.glob(f"*{source_id}*.csv"))
        if not matches:
            raise FileNotFoundError(
                f"No se encontró fichero con id '{source_id}' en {self._data_dir}"
            )
        return matches[0]

    @staticmethod
    def _read_buoy(path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path, sep="\t", skiprows=1, header=0, encoding="latin-1")
        dates = pd.to_datetime(raw[_COL_DATE].str.strip(), format="%Y %m %d %H")
        hs = pd.to_numeric(raw[_COL_HS], errors="coerce")
        df = pd.DataFrame({"hs_inst": hs.values}, index=dates)
        df.replace(_NULL, np.nan, inplace=True)
        return df[df["hs_inst"] > 0].dropna()

    def _reanalysis_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "hs_rean": self._rean.hs,
                "dir_rean": self._rean.directions,
            },
            index=pd.DatetimeIndex(self._rean.timestamps),
        )
