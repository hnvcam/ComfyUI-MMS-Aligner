# ComfyUI-MMS-Aligner — Design

## Goal

A ComfyUI custom node that aligns an audio input with a text transcript and produces an SRT subtitle file with word/line-level timestamps, using Meta's MMS forced-alignment model bundled in `torchaudio` (`torchaudio.pipelines.MMS_FA`) — **no C++ compilation required**.

## Why not `ctc-forced-aligner`?

`ctc-forced-aligner` ships a C++ extension (`align_ops`) that has no prebuilt Windows wheels for Python 3.13 and requires a complete MSVC + Windows SDK install to compile from source. We pivoted to `torchaudio.functional.forced_align` (the same forced-alignment algorithm, built into torchaudio with PyTorch C++ kernels already shipped) and `uroman` (pure-Python romanization). This sacrifices the user-selectable `mms-1b-all` / `mms-300m` choice (only the bundled MMS_FA model is available no-compile), but keeps the multilingual coverage and removes all build-time friction.

## Node: `MMSAligner`

Category: `audio/alignment`

### Inputs

| Name | Type | Default | Notes |
|---|---|---|---|
| `audio` | AUDIO | — | ComfyUI native `{waveform, sample_rate}` |
| `text` | STRING (multiline) | — | Transcript, may include punctuation/newlines |
| `language` | enum | `auto` | `auto`, ~30 common languages, `custom` |
| `custom_language_code` | STRING | `""` | ISO 639-3, used only when `language=custom` |
| `romanize` | BOOLEAN | False | Pass `--romanize` equivalent for non-Latin scripts |
| `segmentation_mode` | enum | `max_chars+punctuation+newlines` | One of: `max_chars`, `punctuation`, `newlines`, `max_chars+punctuation`, `max_chars+punctuation+newlines` |
| `max_chars_per_line` | INT | 42 | 0 = unlimited |
| `output_path` | STRING | `""` | Empty = skip file write |

### Outputs

- `srt` (STRING) — full SRT file content
- `output_path` (STRING) — path written, or `""`

## Architecture

```
ComfyUI-MMS-Aligner/
  __init__.py              # NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
  nodes/
    mms_aligner.py         # MMSAligner ComfyUI node class
  src/
    model_manager.py       # ensure_model(repo_id) -> local path
    aligner.py             # align(audio_path, text, model_path, lang, romanize) -> word timings
    segmenter.py           # group words into SRT segments per mode + max_chars
    srt_writer.py          # segments -> SRT string; optional file write
    lang_detect.py         # detect_language_iso639_3(text) via lingua
    languages.py           # common-language list, ISO 639-1<->639-3 maps
  requirements.txt
  pyproject.toml
```

Each `src/` module is independently testable and free of ComfyUI imports. The node class is a thin orchestrator.

## Data Flow

1. **Resolve language**
   - `auto` → `lang_detect.detect()` from text → ISO 639-3
   - `custom` → use `custom_language_code` (validated against MMS supported set)
   - other → map dropdown label to ISO 639-3 via `languages.py`
2. **Ensure model** — `model_manager.ensure_model(repo_id)` downloads via `huggingface_hub.snapshot_download` to `ComfyUI/models/mms/<repo_name>` if absent; returns local path.
3. **Prepare audio** — resample `audio.waveform` to 16 kHz mono **in memory** (no temp file).
4. **Align** — `aligner.align()` calls the `ctc_forced_aligner` Python API directly (`load_alignment_model`, `generate_emissions`, `preprocess_text`, `get_alignments`, `get_spans`, `postprocess_results`) with the in-memory tensor, returning word-level `{text, start, end}` entries.
5. **Segment** — `segmenter.segment(words, mode, max_chars)` produces SRT cues.
6. **Emit SRT** — `srt_writer.format(cues)` -> string; if `output_path` set, write file.
7. **Return** `(srt_string, output_path or "")`.

## Segmentation Modes

Given word-level timestamps and the original text:

- `max_chars` — pack words greedily up to `max_chars_per_line`
- `punctuation` — break at `. ! ? , ; :` (configurable list internally) in the source text
- `newlines` — break at `\n` in the source text
- `max_chars+punctuation` — break at punctuation OR when next word would exceed `max_chars`
- `max_chars+punctuation+newlines` — same, with newline as a hard break

`max_chars_per_line=0` disables the char limit; modes that contain `max_chars` then behave as a "no length cap" mode that still respects punctuation/newlines if present.

## Language Detection

`lingua-language-detector` (pure-Python, model-free, high accuracy on short text). Detects ISO 639-1; `languages.py` maps to ISO 639-3 (e.g., `vi` → `vie`, `en` → `eng`). On failure (text too short / mixed), raise with a clear "please set language explicitly" message.

## Model Management

- `ComfyUI/models/mms/` is resolved via `folder_paths.models_dir` (ComfyUI helper) with fallback to `<this-package>/../../models/mms`.
- `ensure_model(repo_id)`:
  - Local dir = `models/mms/<repo_id.replace('/', '__')>` (e.g. `facebook__mms-1b-all`)
  - If exists and non-empty → return it
  - Else `snapshot_download(repo_id, local_dir=..., local_dir_use_symlinks=False)`
- LID model is **not** needed (auto-detect runs on text only).

## Error Handling

| Condition | Behavior |
|---|---|
| `ctc-forced-aligner` not installed | Raise `ImportError` with `pip install ctc-forced-aligner` hint |
| HF download failure | Re-raise with hint about network/`HF_TOKEN`/`HF_HOME` |
| Empty text or empty audio | Raise `ValueError` |
| `language=auto` but lingua can't decide | Raise `ValueError("Could not auto-detect language; please set explicitly")` |
| `custom_language_code` not in MMS-supported set | Raise `ValueError` listing nearest matches |

## Dependencies

`requirements.txt`:
- `uroman` (pure Python)
- `lingua-language-detector` (pure Python)

(`torch`, `torchaudio`, `numpy` are provided by ComfyUI. `torchaudio.pipelines.MMS_FA` and `torchaudio.functional.forced_align` ship with torchaudio.)

## Repo Hygiene

- `README.md` — intro, features, install, model download notes, node reference (all inputs/outputs with descriptions), usage examples, troubleshooting, license.
- `LICENSE` — MIT.
- `pyproject.toml` — package metadata for ComfyUI-Manager discovery.
- Every node input declared with a `tooltip` in its options dict so ComfyUI shows hover help.

## Out of Scope

- Audio-based language identification (would require MMS-LID, ~1 GB extra download).
- Multi-speaker diarization.
- Real-time / streaming alignment.
- Subtitle styling (SSA/ASS) — SRT only.
