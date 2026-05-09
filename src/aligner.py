"""Forced alignment using `facebook/mms-1b-fl102` via transformers + torchaudio.

Pure-Python install path: no C++ compilation. Uses
`transformers.Wav2Vec2ForCTC` with per-language adapters and
`torchaudio.functional.forced_align` for the alignment math.

`uroman` is used as an optional preprocessing step when `romanize=True` —
useful as a fallback if your text has unicode characters that aren't in the
target language's tokenizer vocabulary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torchaudio
import torchaudio.functional as TAF


MMS_SAMPLE_RATE = 16000


@dataclass
class WordTiming:
    text: str
    start: float
    end: float


def _prepare_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Return a (1, T) float32 waveform at 16 kHz mono."""
    wav = waveform
    if wav.dim() == 3:
        wav = wav[0]
    if wav.dim() == 2:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dtype != torch.float32:
        wav = wav.float()
    if sample_rate != MMS_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sample_rate, MMS_SAMPLE_RATE)
    return wav.contiguous()


_WORD_STRIP_RE = re.compile(r"^[\W_]+|[\W_]+$", flags=re.UNICODE)


def _maybe_romanize(text: str, language: str | None, uroman_obj) -> str:
    if uroman_obj is None:
        return text
    try:
        return uroman_obj.romanize_string(text, lcode=language or None)
    except TypeError:
        return uroman_obj.romanize_string(text)


def _load_uroman():
    try:
        import uroman as ur
    except ImportError as e:
        raise ImportError(
            "uroman is required for romanization. pip install uroman"
        ) from e
    return ur.Uroman()


def resolve_dtype(precision: str) -> torch.dtype:
    """Map a precision label to a torch.dtype. Raises if fp8 is requested
    but the runtime doesn't expose it."""
    p = (precision or "fp32").lower()
    if p in ("fp32", "float32", "f32"):
        return torch.float32
    if p in ("fp16", "float16", "half", "f16"):
        return torch.float16
    if p in ("bf16", "bfloat16"):
        return torch.bfloat16
    if p in ("fp8", "float8", "f8", "fp8_e4m3"):
        if hasattr(torch, "float8_e4m3fn"):
            return torch.float8_e4m3fn
        raise ValueError(
            "fp8 requires PyTorch 2.1+ with float8 support. "
            "Use bf16 or fp16 instead."
        )
    raise ValueError(f"Unknown precision: {precision!r}")


def _load_model_and_processor(model_path: str, language: str, device: str,
                              dtype: torch.dtype = torch.float32):
    try:
        from transformers import Wav2Vec2ForCTC, AutoProcessor
    except ImportError as e:
        raise ImportError(
            "transformers is required. pip install transformers"
        ) from e

    processor = AutoProcessor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_path,
        target_lang=language,
        ignore_mismatched_sizes=True,
    )
    processor.tokenizer.set_target_lang(language)
    model.load_adapter(language)
    # fp8 weights must be cast after loading; .to(dtype=fp8) on conv/norm
    # layers will fail on most runtimes, so we cast linears only.
    if dtype == getattr(torch, "float8_e4m3fn", None):
        model = model.to(device).eval()
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data = m.weight.data.to(dtype)
    else:
        model = model.to(device=device, dtype=dtype).eval()
    return model, processor


def _tokenize_word(processor, word: str) -> list[int]:
    """Return the tokenizer's char-level ids for a single word."""
    ids = processor.tokenizer(word, add_special_tokens=False).input_ids
    if not isinstance(ids, list):
        ids = list(ids)
    return ids


def align(
    waveform: torch.Tensor,
    sample_rate: int,
    text: str,
    language: str,
    romanize: bool,
    model_path: str,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> list[WordTiming]:
    """Force-align `text` to `waveform`. Returns word-level timings."""
    if not text.strip():
        raise ValueError("Input text is empty.")
    if waveform.numel() == 0:
        raise ValueError("Input audio is empty.")
    if not language:
        raise ValueError("Language code is required (ISO 639-3).")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = _load_model_and_processor(model_path, language, device, dtype=dtype)
    blank_id = model.config.pad_token_id
    if blank_id is None:
        blank_id = 0

    audio = _prepare_waveform(waveform, sample_rate).to(device)
    # Cast audio to the same compute dtype as the model (skip for fp8 where
    # we keep activations in higher precision and only weights are quantised).
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if dtype != torch.float32 and dtype != fp8:
        audio = audio.to(dtype)
    with torch.inference_mode():
        logits = model(audio).logits  # (1, T, V)
        # forced_align expects float32 log_probs
        log_probs = torch.log_softmax(logits.float(), dim=-1)

    uroman_obj = _load_uroman() if romanize else None

    raw_words = text.split()
    word_token_lists: list[list[int]] = []
    kept_words: list[str] = []
    for w in raw_words:
        stripped = _WORD_STRIP_RE.sub("", w)
        if not stripped:
            continue
        normalized = _maybe_romanize(stripped, language, uroman_obj).lower()
        ids = _tokenize_word(processor, normalized)
        # Drop unknown-token ids if the tokenizer exposes one
        unk_id = getattr(processor.tokenizer, "unk_token_id", None)
        if unk_id is not None:
            ids = [i for i in ids if i != unk_id]
        if ids:
            word_token_lists.append(ids)
            kept_words.append(w)

    if not word_token_lists:
        raise ValueError(
            "Transcript produced no alignable tokens after normalization. "
            "Try toggling `romanize`, or verify the language code matches the text."
        )

    flat_targets = [t for toks in word_token_lists for t in toks]
    targets = torch.tensor([flat_targets], dtype=torch.int32, device=device)

    alignments, scores = TAF.forced_align(log_probs, targets, blank=blank_id)
    aligned = alignments[0]
    aligned_scores = scores[0].exp()

    token_spans = TAF.merge_tokens(aligned, aligned_scores)
    if len(token_spans) != len(flat_targets):
        raise RuntimeError(
            f"Forced-alignment span count ({len(token_spans)}) does not match "
            f"target token count ({len(flat_targets)}). The transcript may not "
            f"match the audio."
        )

    num_frames = log_probs.size(1)
    sec_per_frame = (audio.size(-1) / num_frames) / MMS_SAMPLE_RATE

    out: list[WordTiming] = []
    cursor = 0
    for word, toks in zip(kept_words, word_token_lists):
        spans = token_spans[cursor:cursor + len(toks)]
        cursor += len(toks)
        start = spans[0].start * sec_per_frame
        end = spans[-1].end * sec_per_frame
        out.append(WordTiming(text=word, start=float(start), end=float(end)))
    return out
