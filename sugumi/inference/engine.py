"""Dependency-free mock chat engine."""

from collections.abc import Mapping, Sequence


class SugumiEngine:
    """A small mock engine for exercising the chatbot interfaces."""

    def __init__(self, mock: bool = True, checkpoint: str | None = None) -> None:
        if not mock:
            raise NotImplementedError(
                "Real model inference is not implemented; use SugumiEngine(mock=True)."
            )
        if checkpoint is not None:
            raise ValueError("A checkpoint cannot be loaded in mock mode.")

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return a clearly labeled mock response to the latest user message."""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "").strip()
                if content:
                    return f"[Mock response] You said: {content}"
                break
        return "[Mock response] Please provide a non-empty user message."
