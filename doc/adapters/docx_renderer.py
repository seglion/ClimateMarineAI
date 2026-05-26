from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Emu
from docx.text.paragraph import Paragraph

from entities.document import DocContext, DocumentResult
from interfaces.ports import TemplateRenderer

_TEXT_MAP: dict[str, str] = {
    "Anex_number":            "anex_number",
    "Project_ID":             "project_id",
    "Project_type":           "project_type",
    "Zone_id":                "zone_id",
    "Hindcast_Type":          "hindcast_type",
    "Instrumental_Hindcast":  "instrumental_hindcast",
    "Wave_Location":          "wave_location",
    "Source_Id":              "source_id",
    "Wave_lat":               "wave_lat",
    "Wave_long":              "wave_long",
    "Wave_start_date":        "wave_start_date",
    "Wave_end_date":          "wave_end_date",
    "H50":                    "h50",
    "H90":                    "h90",
    "H95":                    "h95",
    "H99":                    "h99",
    "H99_9":                  "h99_9",
    "bm_H2":                  "bm_h2",
    "bm_H2_bcs":              "bm_h2_bcs",
    "bm_H5":                  "bm_h5",
    "bm_H5_bcs":              "bm_h5_bcs",
    "bm_H10":                 "bm_h10",
    "bm_H10_bcs":             "bm_h10_bcs",
    "bm_H100":                "bm_h100",
    "bm_H100_bcs":            "bm_h100_bcs",
    "bm_H200":                "bm_h200",
    "bm_H200_bcs":            "bm_h200_bcs",
    "pot_H2":                 "pot_h2",
    "pot_H2_bcs":             "pot_h2_bcs",
    "pot_H5":                 "pot_h5",
    "pot_H5_bcs":             "pot_h5_bcs",
    "pot_H10":                "pot_h10",
    "pot_H10_bcs":            "pot_h10_bcs",
    "pot_H100":               "pot_h100",
    "pot_H100_bcs":           "pot_h100_bcs",
    "pot_H200":               "pot_h200",
    "pot_H200_bcs":           "pot_h200_bcs",
    "Analisys_Predominant_Wave_LLM": "analisys_predominant_wave_llm",
}

# placeholder → campo Path único en DocContext
_IMAGE_MAP: dict[str, str] = {
    "Mean_wave_regime":       "mean_wave_regime_fig",
    "Extreme_bm_figure":      "extreme_bm_fig",
    "Extreme_pot_figure":     "extreme_pot_fig",
    "BIvariate_Distribution": "bivariate_distribution_fig",
    "Wave_Occurency_pair":    "wave_occurency_pair_fig",
}

# placeholder → (campo Path izquierda, campo Path derecha)
_IMAGE_PAIR_MAP: dict[str, tuple[str, str]] = {
    "FIGURE_WAVE_ROSE_PAIR": ("wave_rose_fig", "extreme_wave_rose_fig"),
}

# placeholder imagen → texto que reemplaza al token en el caption
_CAPTION_MAP: dict[str, str] = {
    "Mean_wave_regime":       "Régimen Medio de Oleaje",
    "Extreme_bm_figure":      "Régimen Extremal de Oleaje – Método Máximos Anuales (GEV)",
    "Extreme_pot_figure":     "Régimen Extremal de Oleaje – Método POT ",
    "BIvariate_Distribution": "Distribución Conjunta Hs-Tp",
    "Wave_Occurency_pair":    "Ocurrencia Conjunta Hs-Tp",
    "FIGURE_WAVE_ROSE_PAIR":  "Rosa de Oleaje Media (izq.) y Extremal (dch.)",
}

_ALL_IMAGE_KEYS = set(_IMAGE_MAP) | set(_IMAGE_PAIR_MAP)
_PLACEHOLDER_RE  = re.compile(r"\{([A-Za-z0-9_]+)\}")

_W_R       = qn("w:r")
_W_DRAWING = qn("w:drawing")
_WP_EXTENT = qn("wp:extent")


class DocxTemplateRenderer(TemplateRenderer):
    """Adapter python-docx.

    Sustitución de texto: concatena todos los runs del párrafo (Word los
    fragmenta por el corrector) y reconstruye el primero con el texto final.

    Sustitución de imagen: el caption contiene {placeholder}. La imagen
    placeholder está en el párrafo ANTERIOR. Se reemplazan los drawings de
    ese párrafo con las imágenes reales, respetando las dimensiones EMU
    originales. El token se elimina del caption.
    """

    def render(
        self,
        template_path: Path,
        context: DocContext,
        output_path: Path,
    ) -> DocumentResult:
        doc = Document(str(template_path))

        text_values = self._text_values(context)
        image_single = self._image_values(context)
        image_pairs  = self._image_pair_values(context)

        # Lista para poder acceder al párrafo anterior por índice
        paragraphs = list(self._all_paragraphs(doc))

        filled  = 0
        skipped: list[str] = []

        for idx, para in enumerate(paragraphs):
            # ── sustitución de texto ──────────────────────────────────────
            n, sk = _replace_text(para, text_values)
            filled  += n
            skipped += sk

            # ── ¿tiene placeholder de imagen en el caption? ───────────────
            ph = _detect_image_ph(para)
            if ph is None:
                continue

            # El párrafo anterior contiene la(s) imagen(es) placeholder
            if idx == 0:
                skipped.append(ph)
                continue

            img_para = paragraphs[idx - 1]
            sizes    = _get_drawing_sizes(img_para)

            if ph in image_pairs:
                left_path, right_path = image_pairs[ph]
                if left_path.exists() and right_path.exists() and len(sizes) >= 2:
                    _replace_drawings(img_para, [
                        (left_path,  sizes[0][0], sizes[0][1]),
                        (right_path, sizes[1][0], sizes[1][1]),
                    ])
                    filled += 1
                else:
                    skipped.append(ph)

            elif ph in image_single:
                img_path = image_single[ph]
                if img_path is not None and img_path.exists() and sizes:
                    _replace_drawings(img_para, [(img_path, sizes[0][0], sizes[0][1])])
                    filled += 1
                else:
                    skipped.append(ph)

            # Reemplazar el token {ph} del caption con el texto descriptivo
            _strip_token(para, ph, _CAPTION_MAP.get(ph, ""))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        return DocumentResult(
            path=output_path,
            placeholders_filled=filled,
            placeholders_skipped=skipped,
        )

    # ── construcción de mapas de valores ─────────────────────────────────────

    @staticmethod
    def _text_values(ctx: DocContext) -> dict[str, str]:
        return {ph: getattr(ctx, field) for ph, field in _TEXT_MAP.items()}

    @staticmethod
    def _image_values(ctx: DocContext) -> dict[str, Path | None]:
        return {ph: getattr(ctx, field) for ph, field in _IMAGE_MAP.items()}

    @staticmethod
    def _image_pair_values(ctx: DocContext) -> dict[str, tuple[Path, Path]]:
        result: dict[str, tuple[Path, Path]] = {}
        for ph, (f_left, f_right) in _IMAGE_PAIR_MAP.items():
            left  = getattr(ctx, f_left,  None)
            right = getattr(ctx, f_right, None)
            if left is not None and right is not None:
                result[ph] = (Path(left), Path(right))
        return result

    @staticmethod
    def _all_paragraphs(doc: Document) -> Generator[Paragraph, None, None]:
        yield from doc.paragraphs
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    yield from cell.paragraphs
        for section in doc.sections:
            if section.header:
                yield from section.header.paragraphs
            if section.footer:
                yield from section.footer.paragraphs


# ── funciones de manipulación XML ────────────────────────────────────────────

def _replace_text(para: Paragraph, text_values: dict[str, str]) -> tuple[int, list[str]]:
    """Sustituye placeholders de texto concatenando todos los runs."""
    if not para.runs:
        return 0, []
    full = "".join(r.text for r in para.runs)
    if "{" not in full:
        return 0, []

    modified = full
    count    = 0
    skipped  = []

    for ph in _PLACEHOLDER_RE.findall(full):
        token = "{" + ph + "}"
        if ph in text_values:
            modified = modified.replace(token, text_values[ph])
            count += 1
        elif ph not in _ALL_IMAGE_KEYS:
            skipped.append(ph)

    if count:
        para.runs[0].text = modified
        for run in para.runs[1:]:
            run.text = ""

    return count, skipped


def _detect_image_ph(para: Paragraph) -> str | None:
    """Devuelve el nombre del placeholder de imagen del caption, o None."""
    full = "".join(r.text for r in para.runs)
    for ph in _PLACEHOLDER_RE.findall(full):
        if ph in _ALL_IMAGE_KEYS:
            return ph
    return None


def _get_drawing_sizes(para: Paragraph) -> list[tuple[int, int]]:
    """Devuelve lista de (cx, cy) en EMU para cada imagen inline del párrafo."""
    return [
        (int(ext.get("cx", 0)), int(ext.get("cy", 0)))
        for ext in para._p.findall(".//" + _WP_EXTENT)
    ]


def _replace_drawings(
    para: Paragraph,
    new_images: list[tuple[Path, int, int]],
) -> None:
    """Elimina los drawings existentes e inserta nuevas imágenes con los
    mismos tamaños (cx, cy en EMU)."""
    # Eliminar todos los <w:r> que contengan <w:drawing>
    for r_elem in list(para._p.findall(_W_R)):
        if r_elem.find(_W_DRAWING) is not None:
            para._p.remove(r_elem)

    # Insertar nuevas imágenes
    for img_path, cx, cy in new_images:
        run = para.add_run()
        run.add_picture(str(img_path), width=Emu(cx), height=Emu(cy))


def _strip_token(para: Paragraph, token_key: str, replacement: str = "") -> None:
    """Reemplaza {token_key} en el caption con replacement (por defecto cadena vacía)."""
    token = "{" + token_key + "}"
    full  = "".join(r.text for r in para.runs)
    if token not in full:
        return
    modified = full.replace(token, replacement)
    if para.runs:
        para.runs[0].text = modified
        for r in para.runs[1:]:
            r.text = ""
