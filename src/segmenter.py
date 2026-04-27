"""Group word timings into SRT cues per a segmentation mode."""

from __future__ import annotations

from dataclasses import dataclass

from .aligner import WordTiming


SEGMENTATION_MODES = (
    "single_word",
    "max_chars",
    "punctuation",
    "newlines",
    "max_chars+punctuation",
    "max_chars+punctuation+newlines",
)

PUNCT_BREAKS = set(".!?,;:")


@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str


def bridge_gaps(cues: list[Cue], gap_ms: int) -> list[Cue]:
    """Extend each cue's end to the next cue's start if the gap is < gap_ms.

    Larger gaps are preserved (real pauses stay visible as silence).
    """
    if gap_ms <= 0 or len(cues) < 2:
        return cues
    threshold = gap_ms / 1000.0
    for i in range(len(cues) - 1):
        gap = cues[i + 1].start - cues[i].end
        if 0 < gap < threshold:
            cues[i] = Cue(
                index=cues[i].index,
                start=cues[i].start,
                end=cues[i + 1].start,
                text=cues[i].text,
            )
    return cues


def _flush(cues: list[Cue], buf: list[WordTiming]) -> None:
    if not buf:
        return
    text = " ".join(w.text for w in buf).strip()
    if not text:
        buf.clear()
        return
    cues.append(Cue(index=len(cues) + 1, start=buf[0].start, end=buf[-1].end, text=text))
    buf.clear()


def _word_ends_with_punct(text: str) -> bool:
    return bool(text) and text[-1] in PUNCT_BREAKS


def _word_starts_paragraph(raw_text_before: str) -> bool:
    """Heuristic: a newline immediately precedes this word in the raw input."""
    return raw_text_before.endswith("\n")


def segment(
    words: list[WordTiming],
    mode: str,
    max_chars_per_line: int,
    raw_text: str = "",
) -> list[Cue]:
    """Build SRT cues from word timings.

    `raw_text` is the original transcript (used for newline detection).
    `max_chars_per_line` of 0 disables the character cap.
    """
    if mode not in SEGMENTATION_MODES:
        raise ValueError(f"Unknown segmentation mode: {mode}")
    if not words:
        return []

    if mode == "single_word":
        return [
            Cue(index=i + 1, start=w.start, end=w.end, text=w.text.strip())
            for i, w in enumerate(words)
            if w.text.strip()
        ]

    use_max = "max_chars" in mode
    use_punct = "punctuation" in mode
    use_newlines = "newlines" in mode

    # Precompute per-word "newline before" markers by walking the raw text.
    newline_before: list[bool] = [False] * len(words)
    if use_newlines and raw_text:
        cursor = 0
        lower_raw = raw_text
        for i, w in enumerate(words):
            token = w.text.strip()
            if not token:
                continue
            idx = lower_raw.find(token, cursor)
            if idx < 0:
                # Try case-insensitive
                idx = lower_raw.lower().find(token.lower(), cursor)
                if idx < 0:
                    continue
            preceding = lower_raw[cursor:idx]
            if "\n" in preceding and i > 0:
                newline_before[i] = True
            cursor = idx + len(token)

    cues: list[Cue] = []
    buf: list[WordTiming] = []
    cur_len = 0  # running char length of buf joined with spaces

    for i, w in enumerate(words):
        # Hard break BEFORE this word (newline)
        if use_newlines and newline_before[i] and buf:
            _flush(cues, buf)
            cur_len = 0

        # Length check BEFORE adding (greedy max-chars)
        wlen = len(w.text)
        projected = cur_len + (1 if buf else 0) + wlen
        if use_max and max_chars_per_line > 0 and buf and projected > max_chars_per_line:
            _flush(cues, buf)
            cur_len = 0

        buf.append(w)
        cur_len += wlen + (1 if len(buf) > 1 else 0)

        # Hard break AFTER this word (punctuation)
        if use_punct and _word_ends_with_punct(w.text):
            _flush(cues, buf)
            cur_len = 0

    _flush(cues, buf)
    return cues
