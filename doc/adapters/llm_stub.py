from __future__ import annotations

from entities.document import DocContext
from interfaces.ports import LLMGenerator


class LLMStub(LLMGenerator):
    """Stub: devuelve un marcador para los placeholders LLM.

    Sustituir por AnthropicLLMGenerator cuando se integre Claude.
    """

    def generate(self, placeholder: str, context: DocContext) -> str:
        return f"[PENDIENTE — {placeholder}]"
