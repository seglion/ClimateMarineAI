from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from entities.document import DocContext, DocumentResult


class TemplateRenderer(ABC):
    """Renderiza una plantilla .docx sustituyendo placeholders de texto e imagen."""

    @abstractmethod
    def render(
        self,
        template_path: Path,
        context: DocContext,
        output_path: Path,
    ) -> DocumentResult: ...


class LLMGenerator(ABC):
    """Genera texto para placeholders LLM a partir del contexto de análisis."""

    @abstractmethod
    def generate(self, placeholder: str, context: DocContext) -> str: ...
