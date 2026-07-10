"""Transcription: run faster-whisper on a WAV file, output segment JSON."""
import json
import os
from pathlib import Path


def _add_nvidia_lib_paths() -> None:
    """CTranslate2 links against CUDA 12's cuBLAS/cuDNN, but pip resolves torch's
    CUDA 13 libs by default. Point the dynamic linker at the CUDA 12 wheels
    (nvidia-cublas-cu12 / nvidia-cudnn-cu12) so libcublas.so.12 is found."""
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib

        paths = [
            list(nvidia.cublas.lib.__path__)[0],
            list(nvidia.cudnn.lib.__path__)[0],
        ]
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(paths + ([existing] if existing else []))
    except ImportError:
        pass


_add_nvidia_lib_paths()

from faster_whisper import WhisperModel  # noqa: E402

DEFAULT_MODEL_SIZE = "medium"


def transcribe(
    audio_path: Path,
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = "cuda",
    compute_type: str = "float16",
    task: str = "translate",
) -> tuple[list[dict], str]:
    """Transcribe audio_path, returning ({start, end, text} segments, detected language code).

    task="translate" (default) has Whisper translate non-English speech directly to
    English text, so downstream summarization always works in English regardless of
    the source language — Qwen's instruction-following for non-English output
    (e.g. "respond in Turkish") is not reliable at this model size/quantization.
    Pass task="transcribe" to keep the source language instead.
    """
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(audio_path), beam_size=5, task=task)
    segment_list = [
        {"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()}
        for seg in segments
    ]
    return segment_list, info.language


def transcribe_to_json(audio_path: Path, out_path: Path, **kwargs) -> Path:
    """Transcribe audio_path and write the segment list as JSON to out_path."""
    segments, _language = transcribe(audio_path, **kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False))
    return out_path
