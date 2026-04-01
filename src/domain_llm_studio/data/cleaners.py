"""Data cleaning utilities: deduplication, normalization, length filtering."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace into single spaces, strip edges."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def normalize_unicode(text: str) -> str:
    """NFKC normalization for consistent CJK and Latin characters."""
    return unicodedata.normalize("NFKC", text)


def clean_special_chars(text: str) -> str:
    """Remove control characters except newlines and tabs."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def validate_json_output(output_str: str) -> bool:
    """Check if the output is valid JSON."""
    try:
        json.loads(output_str)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def clean_sample(sample: dict, max_input_len: int = 2048, max_output_len: int = 1024) -> dict | None:
    """Clean a single sample. Returns None if sample should be filtered out."""
    input_text = sample.get("input", "")
    output_text = sample.get("output", "")

    input_text = strip_html(input_text)
    input_text = normalize_unicode(input_text)
    input_text = normalize_whitespace(input_text)
    input_text = clean_special_chars(input_text)

    output_text = normalize_unicode(output_text)
    output_text = normalize_whitespace(output_text)
    output_text = clean_special_chars(output_text)

    if not input_text or not output_text:
        return None

    if len(input_text) > max_input_len:
        input_text = input_text[:max_input_len]
    if len(output_text) > max_output_len:
        return None

    task = sample.get("task", "")
    if task in ("fin_summary", "event_extraction", "doc_qa"):
        if not validate_json_output(output_text):
            return None

    return {
        **sample,
        "input": input_text,
        "output": output_text,
    }


def deduplicate(samples: list[dict], key_field: str = "input") -> list[dict]:
    """Remove duplicates based on content hash of the key field."""
    seen: set[str] = set()
    result = []
    for s in samples:
        h = content_hash(s.get(key_field, ""))
        if h not in seen:
            seen.add(h)
            result.append(s)
    return result


def clean_dataset(
    samples: list[dict],
    max_input_len: int = 2048,
    max_output_len: int = 1024,
) -> tuple[list[dict], dict[str, int]]:
    """Clean and deduplicate a dataset. Returns (cleaned_samples, stats)."""
    original_count = len(samples)

    cleaned = []
    for s in samples:
        result = clean_sample(s, max_input_len, max_output_len)
        if result is not None:
            cleaned.append(result)

    after_clean = len(cleaned)
    cleaned = deduplicate(cleaned)
    after_dedup = len(cleaned)

    stats = {
        "original": original_count,
        "after_cleaning": after_clean,
        "after_dedup": after_dedup,
        "removed_by_cleaning": original_count - after_clean,
        "removed_by_dedup": after_clean - after_dedup,
        "retention_rate": after_dedup / original_count if original_count > 0 else 0.0,
    }
    return cleaned, stats
