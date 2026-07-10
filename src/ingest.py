"""Video ingestion: accept a local file path or a URL and return a local video path."""
import re
from pathlib import Path

import yt_dlp

URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def is_url(source: str) -> bool:
    return bool(URL_RE.match(source))


def download_video(url: str, out_dir: Path) -> Path:
    """Download a video via yt-dlp, return the path to the downloaded file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        # Prefer H.264 (avc1): opencv's bundled ffmpeg (used by the visual module for
        # frame grabbing) can't decode AV1/VP9 on this build, even though the system
        # ffmpeg used for audio extraction handles them fine.
        "format": (
            "bestvideo[vcodec^=avc1][height<=720]+bestaudio/"
            "best[vcodec^=avc1][height<=720]/"
            "bestvideo[height<=720]+bestaudio/best[height<=720]/best"
        ),
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info)).with_suffix(".mp4")


def ingest(source: str, out_dir: Path) -> Path:
    """Given a local path or URL, return a local video file path."""
    if is_url(source):
        return download_video(source, out_dir)

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Input video not found: {path}")
    return path
