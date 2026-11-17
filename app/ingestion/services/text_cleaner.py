import re


def clean_text(text: str) -> str:
    """Normalize extracted document text without removing meaningful content."""

    if not text:
        return ""

    # Normalize Windows and old Mac line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null characters sometimes introduced by extraction.
    text = text.replace("\x00", "")

    # Normalize horizontal whitespace while preserving line breaks.
    text = re.sub(r"[^\S\n]+", " ", text)

    # Remove spaces surrounding line breaks.
    text = re.sub(r" *\n *", "\n", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
