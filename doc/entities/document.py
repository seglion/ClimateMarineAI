from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectMeta:
    """Metadatos del proyecto introducidos por el usuario."""
    anex_number: str
    project_id: str
    project_type: str
    zone_id: str
    hindcast_type: str
    instrumental_hindcast: str
    wave_location: str


@dataclass
class DocContext:
    """Contexto completo para renderizar el documento de clima marítimo."""

    # ── Proyecto ──────────────────────────────────────────────────────────────
    anex_number: str
    project_id: str
    project_type: str
    zone_id: str
    hindcast_type: str
    instrumental_hindcast: str
    wave_location: str

    # ── Punto de datos ────────────────────────────────────────────────────────
    source_id: str
    wave_lat: str
    wave_long: str
    wave_start_date: str
    wave_end_date: str

    # ── Régimen medio ─────────────────────────────────────────────────────────
    h50: str
    h90: str
    h95: str
    h99: str
    h99_9: str

    # ── Régimen extremal BM (block maxima / GEV) ──────────────────────────────
    bm_h2: str
    bm_h2_bcs: str
    bm_h5: str
    bm_h5_bcs: str
    bm_h10: str
    bm_h10_bcs: str
    bm_h100: str
    bm_h100_bcs: str
    bm_h200: str
    bm_h200_bcs: str

    # ── Régimen extremal POT (GPD) ────────────────────────────────────────────
    pot_h2: str
    pot_h2_bcs: str
    pot_h5: str
    pot_h5_bcs: str
    pot_h10: str
    pot_h10_bcs: str
    pot_h100: str
    pot_h100_bcs: str
    pot_h200: str
    pot_h200_bcs: str

    # ── Figuras (rutas a imágenes generadas) ──────────────────────────────────
    mean_wave_regime_fig: Path
    extreme_bm_fig: Path
    extreme_pot_fig: Path
    wave_rose_fig: Path           # rosa de oleaje (para FIGURE_WAVE_ROSE_PAIR, izquierda)
    extreme_wave_rose_fig: Path   # rosa de oleaje extremo (FIGURE_WAVE_ROSE_PAIR, derecha)
    bivariate_distribution_fig: Path | None = None
    wave_occurency_pair_fig: Path | None = None

    # ── LLM (rellenado por el use case antes de renderizar) ───────────────────
    analisys_predominant_wave_llm: str = ""


@dataclass
class DocumentResult:
    """Resultado de la generación del documento."""
    path: Path
    placeholders_filled: int
    placeholders_skipped: list[str] = field(default_factory=list)
