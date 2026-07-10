"""Merge: interleave transcript segments and visual notes into one timeline."""


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"[{h:02d}:{m:02d}:{s:02d}]"
    return f"[{m:02d}:{s:02d}]"


def merge_timeline(transcript_segments: list[dict], visual_notes: list[dict] | None = None) -> list[dict]:
    """Combine transcript segments and visual notes into one time-sorted event list.

    Each event: {"start": float, "type": "speech"|"visual", "text": str}
    """
    events = []
    for seg in transcript_segments:
        events.append({"start": seg["start"], "type": "speech", "text": seg["text"]})
    for note in visual_notes or []:
        events.append({"start": note["timestamp"], "type": "visual", "text": note["description"]})
    events.sort(key=lambda e: e["start"])
    return events


def format_timeline(events: list[dict]) -> str:
    """Render a timeline as one line per event, timestamped."""
    lines = []
    for e in events:
        ts = format_timestamp(e["start"])
        if e["type"] == "visual":
            lines.append(f"{ts} (on-screen) {e['text']}")
        else:
            lines.append(f"{ts} {e['text']}")
    return "\n".join(lines)
