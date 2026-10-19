"""Shared chat response checks and diagnostics for failed attempts."""

import logging
from pathlib import Path

from openai.types.chat import ChatCompletion

log = logging.getLogger(__name__)


class EmptyResponseError(ValueError):
    """A completed chat response has no final answer text."""


def response_content(
    response: ChatCompletion, *, max_tokens: int, response_path: Path | None = None,
) -> str:
    """Return the content of one completed, nonempty, non-refused response."""
    if not isinstance(response, ChatCompletion):
        raise ValueError("Chat completion response must be a JSON object.")

    details = (
        f"max_tokens={max_tokens}, usage={response.usage!r}, "
        f"raw response={response_path or 'not saved (enable --debug)'}"
    )
    if len(response.choices) != 1:
        raise ValueError(f"Expected one choice, received {len(response.choices)} ({details}).")
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        raise ValueError(f"Chat completion choice missing message ({details}).")

    finish_reason = getattr(choice, "finish_reason", None)
    details = f"finish_reason={finish_reason!r}, {details}"
    if finish_reason == "length":
        raise ValueError(
            f"Chat completion truncated ({details}). Output token budget exhausted; "
            "increase --max-tokens within the server context limit."
        )
    if finish_reason != "stop":
        raise ValueError(f"Chat completion did not finish normally ({details}).")
    if getattr(message, "refusal", None):
        raise ValueError(f"Chat completion refused: {message.refusal} ({details}).")
    content = getattr(message, "content", None)
    if content is None or (isinstance(content, str) and not content.strip()):
        raise EmptyResponseError(f"Chat completion message content is empty or unsupported ({details}).")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Chat completion message content is empty or unsupported ({details}).")
    return content


def log_response_error(
    operation: str, error: Exception, *, attempt: int, max_attempts: int,
    elapsed_s: float, server: str | None = None,
) -> None:
    """Log any API, parsing, validation or saving error once per failed attempt."""
    log.warning(
        "%s: attempt %d/%d failed after %.1fs%s: %s: %s; cause: %r",
        operation, attempt + 1, max_attempts, elapsed_s,
        f" at {server}" if server else "", type(error).__name__, error, error.__cause__,
        exc_info=log.isEnabledFor(logging.DEBUG),
    )
