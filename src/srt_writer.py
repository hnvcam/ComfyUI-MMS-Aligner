"""Convert SRT cues to an SRT-formatted string and (optionally) write to disk."""

from __future__ import annotations

import re
from pathlib import Path

from .segmenter import Cue


# Common subtitle punctuation. Apostrophes and hyphens kept so words like
# "don't" / "well-known" survive intact.
_PUNCT_CHARS = ".,!?;:\"()[]{}—–…«»“”‘’`"
_PUNCT_RE = re.compile(f"[{re.escape(_PUNCT_CHARS)}]")
_MULTISPACE_RE = re.compile(r"\s+")


def strip_punctuation(text: str) -> str:
    cleaned = _PUNCT_RE.sub("", text)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def cues_to_srt(cues: list[Cue], remove_punctuation: bool = False) -> str:
    parts: list[str] = []
    out_index = 0
    for c in cues:
        text = strip_punctuation(c.text) if remove_punctuation else c.text
        if not text:
            continue
        out_index += 1
        parts.append(str(out_index))
        parts.append(f"{_format_timestamp(c.start)} --> {_format_timestamp(c.end)}")
        parts.append(text)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def write_srt(srt_text: str, output_path: str) -> str:
    """Write `srt_text` to `output_path`. Creates parent dirs. Returns the path."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(srt_text, encoding="utf-8")
    return str(p)


def comfyui_output_dir() -> Path:
    """Return ComfyUI's output directory, falling back to ./output."""
    try:
        import folder_paths  # provided by ComfyUI at runtime
        return Path(folder_paths.get_output_directory())
    except Exception:
        here = Path(__file__).resolve()
        repo_root = here.parent.parent
        if repo_root.parent.name == "custom_nodes":
            return repo_root.parent.parent / "output"
        return repo_root / "output"


def split_cues(cues: list[Cue], split_count: int) -> list[list[Cue]]:
    """Split cues into `split_count` near-equal chunks, re-indexed per chunk."""
    if split_count <= 1 or not cues:
        return [cues]
    n = len(cues)
    k = min(split_count, n)
    base, extra = divmod(n, k)
    chunks: list[list[Cue]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        chunk = cues[start:start + size]
        chunks.append([
            Cue(index=j + 1, start=c.start, end=c.end, text=c.text)
            for j, c in enumerate(chunk)
        ])
        start += size
    return chunks
