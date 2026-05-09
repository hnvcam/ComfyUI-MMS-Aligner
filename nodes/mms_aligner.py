"""ComfyUI node: align audio + transcript into an SRT file using MMS."""

from __future__ import annotations

from pathlib import Path

from ..src import (
    aligner,
    lang_detect,
    languages,
    model_manager,
    segmenter,
    srt_writer,
    whisper_chunker,
)


class MMSAligner:
    """Force-align audio with a transcript and emit an SRT subtitle file.

    Uses `facebook/mms-1b-fl102` via transformers + torchaudio.forced_align.
    Pure Python, no compilation. Weights are auto-downloaded to
    `ComfyUI/models/mms/facebook__mms-1b-fl102/` on first use.
    """

    CATEGORY = "audio/alignment"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("srt", "output_path")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Audio to align. Connect any node that outputs the "
                            "ComfyUI AUDIO type (e.g. LoadAudio). Will be "
                            "resampled to 16 kHz mono in memory."
                        ),
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Transcript to align with the audio. May contain "
                            "punctuation and newlines, both of which can be "
                            "used as segmentation cues for the SRT output."
                        ),
                    },
                ),
                "language": (
                    languages.language_dropdown_choices(),
                    {
                        "default": "auto",
                        "tooltip": (
                            "Language of the transcript. `auto` detects from "
                            "the text. Pick `custom` to enter any ISO 639-3 "
                            "code in `custom_language_code`. The FL102 model "
                            "supports 102 languages — the chosen language "
                            "loads its specific adapter weights and "
                            "tokenizer."
                        ),
                    },
                ),
                "custom_language_code": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "ISO 639-3 code (e.g. `vie`, `eng`, `cmn`). "
                            "Used only when `language` is set to `custom`."
                        ),
                    },
                ),
                "romanize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Romanize the transcript before alignment using "
                            "uroman. Generally NOT needed — the FL102 "
                            "tokenizer is trained on each language's native "
                            "script. Useful as a fallback if you see "
                            "'no alignable tokens' errors with text that "
                            "contains unusual characters."
                        ),
                    },
                ),
                "segmentation_mode": (
                    list(segmenter.SEGMENTATION_MODES),
                    {
                        "default": "max_chars+punctuation+newlines",
                        "tooltip": (
                            "How to split words into SRT subtitle lines:\n"
                            "- single_word: one word per cue (karaoke-style)\n"
                            "- max_chars: pack greedily up to `max_chars_per_line`\n"
                            "- punctuation: break at . ! ? , ; :\n"
                            "- newlines: break at line breaks in the input text\n"
                            "- combinations apply all selected rules"
                        ),
                    },
                ),
                "max_chars_per_line": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 1000,
                        "tooltip": (
                            "Maximum characters per SRT line. 0 disables the "
                            "limit (relevant only when `segmentation_mode` "
                            "includes `max_chars`)."
                        ),
                    },
                ),
                "remove_punctuation": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Strip punctuation (. , ! ? ; : \" ( ) [ ] { } "
                            "— – … « » “ ” ‘ ’) from the SRT output text. "
                            "Apostrophes and hyphens are kept (so \"don't\" "
                            "and \"well-known\" remain intact). Punctuation "
                            "still drives segmentation when "
                            "`segmentation_mode` includes it; only the "
                            "displayed text is cleaned."
                        ),
                    },
                ),
                "output_name": (
                    "STRING",
                    {
                        "default": "subtitles",
                        "tooltip": (
                            "Base file name (no extension) written to ComfyUI's "
                            "output directory. The `.srt` extension is added "
                            "automatically. When `split_count` > 1, files are "
                            "named `<output_name>_<index>.srt`."
                        ),
                    },
                ),
                "gap_ms": (
                    "INT",
                    {
                        "default": 300,
                        "min": 0,
                        "max": 5000,
                        "tooltip": (
                            "Bridge small gaps between consecutive cues. If "
                            "the gap between two cues is smaller than this "
                            "many milliseconds, the previous cue's end is "
                            "snapped to the next cue's start (prevents "
                            "subtitle flicker during fluent speech, "
                            "especially in `single_word` mode). Larger gaps "
                            "are preserved as natural pauses. Set to 0 to "
                            "disable."
                        ),
                    },
                ),
                "split_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 5,
                        "tooltip": (
                            "Number of .srt files to split the output across. "
                            "1 (default) writes a single file. Higher values "
                            "divide cues into near-equal chunks named "
                            "`<output_name>_1.srt`, `<output_name>_2.srt`, ..."
                        ),
                    },
                ),
                "precision": (
                    ["bf16", "fp16", "fp8", "fp32"],
                    {
                        "default": "bf16",
                        "tooltip": (
                            "Compute precision for the MMS model:\n"
                            "- bf16 (default): half the VRAM of fp32, "
                            "wide dynamic range, accurate on Ampere+ GPUs\n"
                            "- fp16: half the VRAM of fp32, may underflow "
                            "on very long audio\n"
                            "- fp8: experimental — quantizes Linear weights "
                            "to float8_e4m3fn, requires PyTorch 2.1+\n"
                            "- fp32: full precision, highest VRAM usage"
                        ),
                    },
                ),
                "chunk_with_whisper": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Split long audio into smaller chunks using "
                            "faster-whisper word-level timestamps to find "
                            "safe split points. Required if MMS OOMs even "
                            "at fp16/bf16 (typically audio > 15-20 min on "
                            "consumer GPUs). Adds a Whisper pass before "
                            "alignment, then merges chunk SRTs."
                        ),
                    },
                ),
                "whisper_model_size": (
                    ["tiny", "base", "small", "medium", "large-v3"],
                    {
                        "default": "base",
                        "tooltip": (
                            "Faster-whisper model size used for chunk-point "
                            "detection (only when chunk_with_whisper=True). "
                            "`base` is usually plenty — we only need rough "
                            "word timings to find anchor points. Larger "
                            "models give more reliable matches in noisy or "
                            "tonal-language audio at the cost of VRAM and "
                            "speed."
                        ),
                    },
                ),
                "chunk_max_seconds": (
                    "INT",
                    {
                        "default": 360,
                        "min": 60,
                        "max": 1800,
                        "tooltip": (
                            "Maximum length per chunk in seconds (only when "
                            "chunk_with_whisper=True). 360s (6 min) is "
                            "conservative and fits comfortably on 16 GB "
                            "GPUs at bf16. Increase if your GPU has more "
                            "VRAM and you want fewer chunks."
                        ),
                    },
                ),
            },
        }

    def _resolve_language(self, language: str, custom_code: str, text: str) -> str:
        if language == "auto":
            return lang_detect.detect_iso639_3(text)
        if language == "custom":
            code = custom_code.strip().lower()
            if not code:
                raise ValueError(
                    "`language` is set to `custom` but `custom_language_code` "
                    "is empty."
                )
            return code
        iso = languages.label_to_iso639_3(language)
        if iso is None:
            raise ValueError(f"Unknown language label: {language}")
        return iso

    def run(
        self,
        audio,
        text: str,
        language: str,
        custom_language_code: str,
        romanize: bool,
        segmentation_mode: str,
        max_chars_per_line: int,
        remove_punctuation: bool,
        gap_ms: int,
        output_name: str,
        split_count: int,
        precision: str = "bf16",
        chunk_with_whisper: bool = False,
        whisper_model_size: str = "base",
        chunk_max_seconds: int = 360,
    ):
        if not text or not text.strip():
            raise ValueError("Input `text` is empty.")

        name = (output_name or "").strip()
        if not name:
            raise ValueError("Input `output_name` is empty.")
        # Strip any user-supplied extension and disallow path separators.
        name = Path(name).name
        if name.lower().endswith(".srt"):
            name = name[:-4]
        if not name:
            raise ValueError("Input `output_name` is empty.")

        split_count = max(1, min(5, int(split_count)))

        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])

        iso_lang = self._resolve_language(language, custom_language_code, text)
        model_path = model_manager.ensure_model()
        dtype = aligner.resolve_dtype(precision)

        if chunk_with_whisper:
            chunks = whisper_chunker.chunk_audio_and_text(
                waveform=waveform,
                sample_rate=sample_rate,
                raw_text=text,
                chunk_max_seconds=float(chunk_max_seconds),
                iso639_3_language=iso_lang,
                whisper_model_size=whisper_model_size,
            )
            words = []
            for ch in chunks:
                sub_wav = whisper_chunker.slice_waveform(
                    waveform, sample_rate, ch.audio_start_sec, ch.audio_end_sec,
                )
                sub_words = aligner.align(
                    waveform=sub_wav,
                    sample_rate=sample_rate,
                    text=ch.text,
                    language=iso_lang,
                    romanize=romanize,
                    model_path=model_path,
                    dtype=dtype,
                )
                for w in sub_words:
                    w.start += ch.audio_start_sec
                    w.end += ch.audio_start_sec
                words.extend(sub_words)
        else:
            words = aligner.align(
                waveform=waveform,
                sample_rate=sample_rate,
                text=text,
                language=iso_lang,
                romanize=romanize,
                model_path=model_path,
                dtype=dtype,
            )

        cues = segmenter.segment(
            words=words,
            mode=segmentation_mode,
            max_chars_per_line=max_chars_per_line,
            raw_text=text,
        )
        cues = segmenter.bridge_gaps(cues, gap_ms)
        srt_text = srt_writer.cues_to_srt(cues, remove_punctuation=remove_punctuation)

        out_dir = srt_writer.comfyui_output_dir()
        chunks = srt_writer.split_cues(cues, split_count)
        written: list[str] = []
        if len(chunks) <= 1:
            target = out_dir / f"{name}.srt"
            written.append(srt_writer.write_srt(srt_text, str(target)))
        else:
            for i, chunk in enumerate(chunks, start=1):
                chunk_text = srt_writer.cues_to_srt(
                    chunk, remove_punctuation=remove_punctuation
                )
                target = out_dir / f"{name}_{i}.srt"
                written.append(srt_writer.write_srt(chunk_text, str(target)))

        return (srt_text, "\n".join(written))
