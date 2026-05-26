from __future__ import annotations

from pathlib import Path

from entities.document import DocContext, DocumentResult, ProjectMeta
from interfaces.ports import LLMGenerator, TemplateRenderer

# Import cruzado permitido: el use case orquesta datos del slice wave
from use_cases.analyze_waves import WaveAnalysisResult


_LLM_PLACEHOLDERS = ("Analisys_Predominant_Wave_LLM",)


class BuildDoc:
    """Use case: construye el documento de clima marítimo.

    Recibe WaveAnalysisResult (slice wave) + ProjectMeta (usuario),
    mapea al DocContext, rellena LLM si hay generador, y renderiza.
    """

    def __init__(
        self,
        renderer: TemplateRenderer,
        llm: LLMGenerator | None = None,
    ) -> None:
        self._renderer = renderer
        self._llm = llm

    def execute(
        self,
        meta: ProjectMeta,
        wave: WaveAnalysisResult,
        template_path: Path,
        output_path: Path,
    ) -> DocumentResult:
        context = self._build_context(meta, wave)

        if self._llm is not None:
            for ph in _LLM_PLACEHOLDERS:
                text = self._llm.generate(ph, context)
                setattr(context, ph.lower(), text)

        return self._renderer.render(template_path, context, output_path)

    # ── mapeo WaveAnalysisResult → DocContext ─────────────────────────────────

    @staticmethod
    def _build_context(meta: ProjectMeta, wave: WaveAnalysisResult) -> DocContext:
        s   = wave.series
        mr  = wave.mean_regime
        bm  = wave.extreme_regime_bm
        pot = wave.extreme_regime_pot

        def fmt_hs(regime, tr: int) -> tuple[str, str]:
            for t, hs, lo, hi in regime.estadisticos:
                if t == tr:
                    return f"{hs:.2f}", f"[{lo:.2f} - {hi:.2f}]"
            return "N/D", "N/D"

        def perc(p: float) -> str:
            return f"{mr.hs_at(p):.2f}"

        bm_h2,   bm_h2_bcs   = fmt_hs(bm, 2)
        bm_h5,   bm_h5_bcs   = fmt_hs(bm, 5)
        bm_h10,  bm_h10_bcs  = fmt_hs(bm, 10)
        bm_h100, bm_h100_bcs = fmt_hs(bm, 100)
        bm_h200, bm_h200_bcs = fmt_hs(bm, 200)

        pot_h2,   pot_h2_bcs   = fmt_hs(pot, 2)
        pot_h5,   pot_h5_bcs   = fmt_hs(pot, 5)
        pot_h10,  pot_h10_bcs  = fmt_hs(pot, 10)
        pot_h100, pot_h100_bcs = fmt_hs(pot, 100)
        pot_h200, pot_h200_bcs = fmt_hs(pot, 200)

        figs = wave.figures

        return DocContext(
            # Proyecto
            anex_number=meta.anex_number,
            project_id=meta.project_id,
            project_type=meta.project_type,
            zone_id=meta.zone_id,
            hindcast_type=meta.hindcast_type,
            instrumental_hindcast=meta.instrumental_hindcast,
            wave_location=meta.wave_location,
            # Serie
            source_id=s.source_id,
            wave_lat=f"{s.latitude:.4f}",
            wave_long=f"{s.longitude:.4f}",
            wave_start_date=f"{s.start:%d/%m/%Y}",
            wave_end_date=f"{s.end:%d/%m/%Y}",
            # Régimen medio
            h50=perc(50), h90=perc(90), h95=perc(95), h99=perc(99), h99_9=perc(99.9),
            # BM
            bm_h2=bm_h2,   bm_h2_bcs=bm_h2_bcs,
            bm_h5=bm_h5,   bm_h5_bcs=bm_h5_bcs,
            bm_h10=bm_h10, bm_h10_bcs=bm_h10_bcs,
            bm_h100=bm_h100, bm_h100_bcs=bm_h100_bcs,
            bm_h200=bm_h200, bm_h200_bcs=bm_h200_bcs,
            # POT
            pot_h2=pot_h2,   pot_h2_bcs=pot_h2_bcs,
            pot_h5=pot_h5,   pot_h5_bcs=pot_h5_bcs,
            pot_h10=pot_h10, pot_h10_bcs=pot_h10_bcs,
            pot_h100=pot_h100, pot_h100_bcs=pot_h100_bcs,
            pot_h200=pot_h200, pot_h200_bcs=pot_h200_bcs,
            # Figuras
            mean_wave_regime_fig=Path(figs["mean_regime"]),
            extreme_bm_fig=Path(figs["extreme_regime_bm"]),
            extreme_pot_fig=Path(figs["extreme_regime_pot"]),
            wave_rose_fig=Path(figs["wave_rose"]),
            extreme_wave_rose_fig=Path(figs["extreme_wave_rose"]),
        )
