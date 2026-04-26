"""Common languages list and ISO 639-1 <-> 639-3 maps.

MMS uses ISO 639-3 codes. The dropdown in the node shows human-readable labels;
this module maps labels and ISO 639-1 (used by lingua) to ISO 639-3.
"""

from __future__ import annotations

# Dropdown label -> ISO 639-3 code (used by MMS / ctc-forced-aligner)
COMMON_LANGUAGES: dict[str, str] = {
    "English": "eng",
    "Vietnamese": "vie",
    "Chinese (Mandarin)": "cmn",
    "Spanish": "spa",
    "French": "fra",
    "German": "deu",
    "Italian": "ita",
    "Portuguese": "por",
    "Russian": "rus",
    "Japanese": "jpn",
    "Korean": "kor",
    "Arabic": "ara",
    "Hindi": "hin",
    "Bengali": "ben",
    "Indonesian": "ind",
    "Malay": "zlm",
    "Thai": "tha",
    "Turkish": "tur",
    "Polish": "pol",
    "Dutch": "nld",
    "Swedish": "swe",
    "Finnish": "fin",
    "Greek": "ell",
    "Czech": "ces",
    "Ukrainian": "ukr",
    "Hebrew": "heb",
    "Persian": "fas",
    "Urdu": "urd",
    "Tamil": "tam",
    "Telugu": "tel",
    "Filipino (Tagalog)": "tgl",
}

# ISO 639-1 -> ISO 639-3 (lingua returns 639-1; we pass 639-3 to MMS)
ISO_639_1_TO_3: dict[str, str] = {
    "en": "eng", "vi": "vie", "zh": "cmn", "es": "spa", "fr": "fra",
    "de": "deu", "it": "ita", "pt": "por", "ru": "rus", "ja": "jpn",
    "ko": "kor", "ar": "ara", "hi": "hin", "bn": "ben", "id": "ind",
    "ms": "zlm", "th": "tha", "tr": "tur", "pl": "pol", "nl": "nld",
    "sv": "swe", "fi": "fin", "el": "ell", "cs": "ces", "uk": "ukr",
    "he": "heb", "fa": "fas", "ur": "urd", "ta": "tam", "te": "tel",
    "tl": "tgl",
}


def language_dropdown_choices() -> list[str]:
    """Return the ordered list of options for the `language` dropdown."""
    return ["auto", *COMMON_LANGUAGES.keys(), "custom"]


def label_to_iso639_3(label: str) -> str | None:
    """Map a dropdown label to its ISO 639-3 code, or None if not a common label."""
    return COMMON_LANGUAGES.get(label)
