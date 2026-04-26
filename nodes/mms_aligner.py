"""ComfyUI node: align audio + transcript into an SRT file using MMS."""

from __future__ import annotations

from ..src import aligner, lang_detect, languages, model_manager, segmenter, srt_writer


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
                "output_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Optional path to write the .srt file. Leave "
                            "empty to skip writing to disk; the SRT is "
                            "always returned as a string output."
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
        output_path: str,
    ):
        if not text or not text.strip():
            raise ValueError("Input `text` is empty.")

        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])

        iso_lang = self._resolve_language(language, custom_language_code, text)
        model_path = model_manager.ensure_model()

        words = aligner.align(
            waveform=waveform,
            sample_rate=sample_rate,
            text=text,
            language=iso_lang,
            romanize=romanize,
            model_path=model_path,
        )

        cues = segmenter.segment(
            words=words,
            mode=segmentation_mode,
            max_chars_per_line=max_chars_per_line,
            raw_text=text,
        )
        srt_text = srt_writer.cues_to_srt(cues, remove_punctuation=remove_punctuation)

        written_path = ""
        if output_path and output_path.strip():
            written_path = srt_writer.write_srt(srt_text, output_path.strip())

        return (srt_text, written_path)
