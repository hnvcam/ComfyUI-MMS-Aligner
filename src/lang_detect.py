"""Text-based language detection using lingua-language-detector."""

from __future__ import annotations

from .languages import ISO_639_1_TO_3


_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        try:
            from lingua import LanguageDetectorBuilder
        except ImportError as e:
            raise ImportError(
                "lingua-language-detector is required for auto language detection. "
                "Install with: pip install lingua-language-detector"
            ) from e
        _detector = LanguageDetectorBuilder.from_all_languages().build()
    return _detector


def detect_iso639_3(text: str) -> str:
    """Detect language of `text`, return ISO 639-3 code.

    Raises ValueError if detection fails or the detected language has no
    ISO 639-3 mapping in our table.
    """
    if not text or not text.strip():
        raise ValueError("Cannot detect language from empty text.")

    detector = _get_detector()
    lang = detector.detect_language_of(text)
    if lang is None:
        raise ValueError(
            "Could not auto-detect language from text. "
            "Please set the language explicitly."
        )

    iso_1 = lang.iso_code_639_1.name.lower()
    iso_3 = ISO_639_1_TO_3.get(iso_1)
    if iso_3 is None:
        # Fallback: lingua exposes 639-3 directly too
        try:
            iso_3 = lang.iso_code_639_3.name.lower()
        except Exception:
            raise ValueError(
                f"Detected language '{iso_1}' has no ISO 639-3 mapping. "
                "Please set the language explicitly."
            )
    return iso_3
