"""
Text preprocessing and cleaning utilities.
"""

import re
import string
from typing import List


def clean_text(text: str) -> str:
    """
    Clean extracted text: remove noise, fix spacing, strip junk.
    """
    if not text:
        return ""

    # Fix encoding artifacts
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    # Remove non-printable characters
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def truncate_text(text: str, max_chars: int = 5000) -> str:
    """Truncate text to max_chars, ending at a sentence boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1]
    return truncated + "..."


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Simple keyword extraction based on word frequency (no ML)."""
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "is", "was", "are",
        "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "shall",
        "can", "this", "that", "these", "those", "it", "its", "they",
        "their", "we", "our", "you", "your", "he", "she", "his", "her",
    }

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    freq = {}
    for word in words:
        if word not in stop_words:
            freq[word] = freq.get(word, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]
