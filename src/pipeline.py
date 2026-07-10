"""CLI entrypoint: video/URL in, summary out.

Usage:
    python -m src.pipeline --input <url_or_path> [--visual on|off]
"""
import argparse
import json
import time
from pathlib import Path

from src.audio import extract_audio
from src.chunk import chunk_to_text
from src.ingest import ingest
from src.merge import merge_timeline
from src.summarize import DEFAULT_MODEL as DEFAULT_SUMMARY_MODEL
from src.summarize import summarize_chunks
from src.transcribe import DEFAULT_MODEL_SIZE as DEFAULT_WHISPER_MODEL
from src.transcribe import transcribe
from src.visual import analyze_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a video with local models only.")
    parser.add_argument("--input", required=True, help="Local video path or URL")
    parser.add_argument("--visual", choices=["on", "off"], default="off",
                         help="Enable the visual module (scene detection + OCR/captioning)")
    parser.add_argument("--visual-mode", choices=["ocr", "caption"], default="ocr",
                         help="Visual analysis mode, only used when --visual on")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL,
                         help="faster-whisper model size (default: %(default)s)")
    parser.add_argument("--summary-model", default=DEFAULT_SUMMARY_MODEL,
                         help="Ollama model for summarization (default: %(default)s)")
    parser.add_argument("--output-dir", default="outputs",
                         help="Base output directory (default: %(default)s)")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Ingesting {args.input!r}...")
    video_path = ingest(args.input, run_dir / "video")

    print("[2/6] Extracting audio...")
    wav_path = extract_audio(video_path, run_dir / "audio.wav")

    print(f"[3/6] Transcribing (faster-whisper: {args.whisper_model})...")
    segments, transcript_language = transcribe(wav_path, model_size=args.whisper_model)
    print(f"      Detected language: {transcript_language}")
    (run_dir / "transcript.json").write_text(json.dumps(segments, indent=2, ensure_ascii=False))

    visual_notes = None
    if args.visual == "on":
        print(f"[4/6] Analyzing visuals (mode: {args.visual_mode})...")
        visual_notes = analyze_video(video_path, mode=args.visual_mode)
        (run_dir / "visual.json").write_text(json.dumps(visual_notes, indent=2, ensure_ascii=False))
    else:
        print("[4/6] Visual module disabled, skipping.")

    print("[5/6] Merging + chunking...")
    events = merge_timeline(segments, visual_notes)
    chunks = chunk_to_text(events)
    print(f"      {len(chunks)} chunk(s) from {len(events)} event(s)")

    print(f"[6/6] Summarizing (Ollama: {args.summary_model})...")
    result = summarize_chunks(chunks, model=args.summary_model)
    (run_dir / "chunk_summaries.json").write_text(
        json.dumps(result["chunk_summaries"], indent=2, ensure_ascii=False)
    )
    (run_dir / "summary.txt").write_text(result["final_summary"])

    print(f"\nDone. Outputs saved to {run_dir}/")
    print("\n=== Summary ===")
    print(result["final_summary"])

    return run_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
