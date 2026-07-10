"""Audio extraction: convert a video file to 16kHz mono WAV via ffmpeg."""
import subprocess
from pathlib import Path


def extract_audio(video_path: Path, out_path: Path) -> Path:
    """Extract a 16kHz mono WAV from video_path, writing to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract audio:\n{result.stderr}")
    return out_path
