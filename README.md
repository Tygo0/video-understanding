# Video Understanding

Turn a video (URL or local file) into a written summary — entirely with local,
open-source models. No hosted LLM APIs; everything runs on-device.

## Pipeline

```
video/URL --> audio (ffmpeg) --> transcript (faster-whisper)
                               \
                                +--> visual notes (PySceneDetect + OCR/captioning) [optional]
                               /
        merge (timeline) --> chunk (~1800 tokens) --> summarize (Ollama, map-reduce) --> translate [optional]
```

1. **Ingest** — accept a local video path or a URL (downloaded via `yt-dlp`)
2. **Audio extraction** — 16kHz mono WAV via `ffmpeg`
3. **Transcription** — `faster-whisper` (medium, float16, GPU), translating non-English speech directly to English text → `{start, end, text}` segments
4. **Visual module** *(optional)* — scene detection (`PySceneDetect`) + keyframe OCR (`EasyOCR`) or captioning (`moondream2`)
5. **Merge + chunk** — interleave transcript and visual notes by timestamp, split into ~1800-token chunks on segment boundaries
6. **Summarize** — map-reduce summarization via a local Ollama model (`qwen2.5:7b-instruct-q4_K_M`), English only
7. **Translate** *(optional)* — translate the final summary into another language via a dedicated model (`NLLB-200`)

## Requirements

- Python 3.x, `ffmpeg`, [Ollama](https://ollama.com) running locally
- An NVIDIA GPU is used where possible (transcription); the visual module falls
  back to CPU if no compatible CUDA-enabled torch build is available for your
  Python version/driver — see [Known limitations](#known-limitations).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

ollama pull qwen2.5:7b-instruct-q4_K_M   # or qwen2.5:3b-instruct if VRAM-constrained
ollama serve                              # if not already running
```

Verify the environment is ready:

```bash
python scripts/check_env.py
```

This checks `ffmpeg`, GPU/CUDA visibility, and that Ollama is reachable with a model pulled.

## Usage

```bash
python -m src.pipeline --input <url_or_local_path> [--visual on|off]
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Local video path or URL |
| `--visual` | `off` | Enable scene detection + OCR/captioning |
| `--visual-mode` | `ocr` | `ocr` (EasyOCR) or `caption` (moondream2), only used with `--visual on` |
| `--whisper-model` | `medium` | `faster-whisper` model size |
| `--summary-model` | `qwen2.5:7b-instruct-q4_K_M` | Ollama model for summarization |
| `--target-language` | `en` | Translate the final summary into this language (e.g. `tr`); `en` skips translation |
| `--output-dir` | `outputs` | Base directory for run outputs |

Each run writes to `outputs/<run_id>/`:

- `video/` — downloaded or copied source video
- `audio.wav` — extracted audio
- `transcript.json` — `{start, end, text}` segments (English, translated from the source language if needed)
- `visual.json` — `{timestamp, description}` notes (if `--visual on`)
- `chunk_summaries.json` — per-chunk summaries (English)
- `summary.txt` — final overall summary (English)
- `summary_<lang>.txt` — translated summary (if `--target-language` is set to something other than `en`)

## Project layout

```
src/
├── ingest.py       # local path or URL -> local video file
├── audio.py        # video -> 16kHz mono WAV
├── transcribe.py    # WAV -> transcript segments (faster-whisper)
├── visual.py         # video -> scene/keyframe OCR or captions
├── merge.py           # transcript + visual notes -> one timestamped timeline
├── chunk.py             # timeline -> token-sized chunks
├── summarize.py          # chunks -> map-reduce summary (Ollama), English only
├── translate.py            # English summary -> target language (NLLB-200), optional
└── pipeline.py               # CLI entrypoint wiring all of the above
scripts/
└── check_env.py    # environment acceptance check (ffmpeg, GPU, Ollama)
```

## Known limitations

- **Visual module runs on CPU.** No CUDA-enabled PyTorch wheel currently exists
  for Python 3.14 compatible with driver versions capped at CUDA 12.5, so
  EasyOCR/moondream2 fall back to CPU. Transcription is unaffected (uses
  `ctranslate2`, not torch) and still runs on GPU. Since visual analysis only
  touches sparse scene-change keyframes rather than every frame, this stays
  within the overall time budget.
- **Singing/lyrics transcription is weaker than speech.** Whisper-family models
  can lock into repetitive hallucination loops on heavily processed or layered
  vocals — treat lyric transcripts with more scrutiny than spoken-word content.
- **Scene detection threshold is tuned for slide/text content** (low-contrast,
  mostly-static backgrounds). Real-world footage with lots of camera motion may
  need a higher `ContentDetector` threshold than the current default.
- **Summarization only works reliably in English.** `qwen2.5:7b-instruct-q4_K_M`'s
  instruction-following for non-English output is unreliable at this size/
  quantization — it drifted into the wrong language entirely when asked to
  summarize non-English text, and produced real mistranslation errors even when
  asked to just translate. Non-English speech is translated to English at the
  transcription stage (`faster-whisper`'s `task="translate"`) so summarization
  always has reliable English input; `--target-language` then translates the
  final summary via a dedicated model (NLLB-200) instead of the LLM.
