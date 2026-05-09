"""Chunk a long (audio, transcript) pair into smaller (audio, text) pairs by
running Whisper to get rough word-level timestamps, then fuzzy-matching
against the raw transcript to find safe split points.

Used when MMS would OOM on a long audio file. Each returned chunk is small
enough to fit in VRAM, and the audio + text boundaries are guaranteed to
correspond to the same word (so MMS forced-alignment will succeed on each
chunk independently)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np
import torch


@dataclass
class Chunk:
    audio_start_sec: float
    audio_end_sec: float
    text: str
    raw_word_start: int  # inclusive index into raw_text.split()
    raw_word_end: int    # exclusive


_NORM_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _normalize(word: str) -> str:
    return _NORM_RE.sub("", word.lower().strip())


# ISO 639-3 (used by MMS) -> ISO 639-1 (used by Whisper)
_ISO3_TO_ISO1 = {
    "vie": "vi", "eng": "en", "cmn": "zh", "fra": "fr", "deu": "de",
    "spa": "es", "ita": "it", "jpn": "ja", "kor": "ko", "rus": "ru",
    "por": "pt", "ara": "ar", "tur": "tr", "tha": "th", "ind": "id",
    "msa": "ms", "nld": "nl", "swe": "sv", "pol": "pl", "ces": "cs",
    "fin": "fi", "ell": "el", "ukr": "uk", "ron": "ro", "hun": "hu",
    "dan": "da", "nor": "no", "heb": "he", "hin": "hi", "ben": "bn",
    "tam": "ta", "tel": "te", "mar": "mr", "fas": "fa", "urd": "ur",
}


def iso639_3_to_whisper(code: str) -> str | None:
    return _ISO3_TO_ISO1.get(code.lower())


def _waveform_to_mono16k_numpy(waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
    import torchaudio
    wav = waveform
    if wav.dim() == 3:
        wav = wav[0]
    if wav.dim() == 2:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dtype != torch.float32:
        wav = wav.float()
    if sample_rate != 16000:
        wav = torchaudio.functional.resample(wav, sample_rate, 16000)
    return wav.squeeze(0).cpu().numpy()


def _whisper_words(audio_np: np.ndarray, language: str | None,
                   model_size: str) -> list[tuple[str, float, float]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ImportError(
            "faster-whisper is required for chunked alignment. "
            "Install with: pip install faster-whisper"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "int8_float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute)
    segments, _info = model.transcribe(
        audio_np,
        language=language,
        word_timestamps=True,
        beam_size=1,
        vad_filter=False,
    )
    words: list[tuple[str, float, float]] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            txt = (w.word or "").strip()
            if not txt:
                continue
            words.append((txt, float(w.start), float(w.end)))
    # Free the whisper model immediately so MMS gets the VRAM back.
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return words


def chunk_audio_and_text(
    waveform: torch.Tensor,
    sample_rate: int,
    raw_text: str,
    chunk_max_seconds: float,
    iso639_3_language: str,
    whisper_model_size: str = "base",
    min_anchor_run: int = 3,
    search_window_sec: float = 60.0,
) -> list[Chunk]:
    """Split (audio, raw_text) into chunks no longer than chunk_max_seconds.

    Returns a list of Chunk records. Each chunk's audio_start/end_sec is in
    the original audio's time frame. Each chunk's text is a slice of
    raw_text by word boundary.

    Algorithm:
    1. Run faster-whisper on the full audio → list of (word, start, end).
    2. Build SequenceMatcher between normalized whisper words and normalized
       raw words. Get matching blocks.
    3. For each target time T (chunk_max_seconds intervals), pick the
       longest matching-block word whose whisper-time is closest to T. Use
       its raw-word index as the text split position.
    4. Each chunk has a confident anchor at both ends (or the audio edge).

    Raises if no confident anchor can be found near a target time, which
    indicates the whisper transcription diverged badly from the raw text.
    """
    raw_words = raw_text.split()
    if not raw_words:
        raise ValueError("raw_text has no words")

    audio_np = _waveform_to_mono16k_numpy(waveform, sample_rate)
    total_dur = len(audio_np) / 16000.0

    if total_dur <= chunk_max_seconds:
        return [Chunk(0.0, total_dur, raw_text, 0, len(raw_words))]

    whisper_lang = iso639_3_to_whisper(iso639_3_language)
    whisper_pairs = _whisper_words(audio_np, language=whisper_lang,
                                   model_size=whisper_model_size)
    if not whisper_pairs:
        raise RuntimeError("Whisper produced no word timings on this audio.")

    norm_raw = [_normalize(w) for w in raw_words]
    norm_wh = [_normalize(w) for w, _, _ in whisper_pairs]

    sm = SequenceMatcher(a=norm_wh, b=norm_raw, autojunk=False)
    matches = [m for m in sm.get_matching_blocks() if m.size >= min_anchor_run]

    boundaries: list[tuple[float, int]] = []  # (split_time, raw_word_idx)
    used_raw = 0
    used_time = 0.0
    target = chunk_max_seconds

    while target < total_dur - 5.0:
        best = None  # (delta, time, raw_idx, wh_idx)
        for m in matches:
            for k in range(m.size):
                wh_idx = m.a + k
                raw_idx = m.b + k
                if raw_idx <= used_raw:
                    continue
                t = whisper_pairs[wh_idx][2]  # use word END time as split point
                if t <= used_time + 1.0:
                    continue
                if abs(t - target) > search_window_sec:
                    continue
                delta = abs(t - target)
                cand = (delta, t, raw_idx, wh_idx)
                if best is None or cand < best:
                    best = cand
        if best is None:
            raise RuntimeError(
                f"Could not find a confident whisper-matched anchor near "
                f"{target:.1f}s (search window ±{search_window_sec:.0f}s). "
                f"Whisper transcription may diverge too much from the raw "
                f"text. Try a larger whisper_model_size, increase "
                f"chunk_max_seconds, or disable chunking."
            )
        _, t, raw_idx, _ = best
        boundaries.append((t, raw_idx))
        used_time = t
        used_raw = raw_idx
        target = t + chunk_max_seconds

    chunks: list[Chunk] = []
    prev_time = 0.0
    prev_raw = 0
    for (t, raw_idx) in boundaries:
        chunks.append(Chunk(
            audio_start_sec=prev_time,
            audio_end_sec=t,
            text=" ".join(raw_words[prev_raw:raw_idx]),
            raw_word_start=prev_raw,
            raw_word_end=raw_idx,
        ))
        prev_time = t
        prev_raw = raw_idx
    chunks.append(Chunk(
        audio_start_sec=prev_time,
        audio_end_sec=total_dur,
        text=" ".join(raw_words[prev_raw:]),
        raw_word_start=prev_raw,
        raw_word_end=len(raw_words),
    ))
    return chunks


def slice_waveform(waveform: torch.Tensor, sample_rate: int,
                   start_sec: float, end_sec: float) -> torch.Tensor:
    """Slice a waveform tensor by time. Returns the same dim layout."""
    s = max(0, int(start_sec * sample_rate))
    e = int(end_sec * sample_rate)
    if waveform.dim() == 3:
        return waveform[..., s:e].contiguous()
    if waveform.dim() == 2:
        return waveform[..., s:e].contiguous()
    return waveform[s:e].contiguous()
