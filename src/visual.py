"""Visual module: scene detection, keyframe extraction, and OCR/captioning.

Output is a JSON-serializable list of {timestamp, description}.
"""
import json
from pathlib import Path

import cv2
from scenedetect import detect, ContentDetector

DEFAULT_MOONDREAM_MODEL = "vikhyatk/moondream2"


def detect_keyframes(video_path: Path, threshold: float = 1.0) -> list[float]:
    """Detect scene changes, return one representative timestamp (scene midpoint) per scene.

    threshold is much lower than PySceneDetect's default (27.0), tuned for
    slide/text content where a cut only changes a small fraction of frame
    pixels against a mostly-static background.
    """
    scene_list = detect(str(video_path), ContentDetector(threshold=threshold))
    if not scene_list:
        return [0.0]
    timestamps = []
    for start, end in scene_list:
        mid = (start.get_seconds() + end.get_seconds()) / 2
        timestamps.append(round(mid, 2))
    return timestamps


def grab_frame(video_path: Path, timestamp: float):
    """Return the BGR frame at timestamp (seconds), or None if it can't be read."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


class OcrReader:
    """Lazily-initialized EasyOCR reader (avoids paying model load cost if unused)."""

    _reader = None

    @classmethod
    def get(cls):
        if cls._reader is None:
            import easyocr

            cls._reader = easyocr.Reader(["en"], gpu=True)
        return cls._reader


def ocr_frame(frame) -> str:
    """Run OCR on a frame, return concatenated detected text (empty string if none)."""
    reader = OcrReader.get()
    results = reader.readtext(frame, detail=0)
    return " ".join(results).strip()


class Captioner:
    """Lazily-initialized moondream2 captioning model."""

    _model = None
    _tokenizer = None

    @classmethod
    def get(cls):
        if cls._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            cls._model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_MOONDREAM_MODEL, trust_remote_code=True, device_map={"": "cuda"}
            )
            cls._tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MOONDREAM_MODEL)
        return cls._model, cls._tokenizer


def caption_frame(frame) -> str:
    """Caption a frame with moondream2, return the description (empty string on failure)."""
    from PIL import Image

    model, tokenizer = Captioner.get()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    enc = model.encode_image(image)
    return model.answer_question(enc, "Describe this image.", tokenizer).strip()


def analyze_video(video_path: Path, mode: str = "ocr") -> list[dict]:
    """Detect scenes, extract a keyframe per scene, and describe each via OCR or captioning.

    mode: "ocr" (EasyOCR, default — good for slides/on-screen text) or
          "caption" (moondream2 — general visual description).
    """
    if mode not in ("ocr", "caption"):
        raise ValueError(f"Unknown mode: {mode!r}")

    describe = ocr_frame if mode == "ocr" else caption_frame

    notes = []
    for ts in detect_keyframes(video_path):
        frame = grab_frame(video_path, ts)
        if frame is None:
            continue
        description = describe(frame)
        if description:
            notes.append({"timestamp": ts, "description": description})
    return notes


def analyze_to_json(video_path: Path, out_path: Path, mode: str = "ocr") -> Path:
    notes = analyze_video(video_path, mode=mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(notes, indent=2, ensure_ascii=False))
    return out_path
