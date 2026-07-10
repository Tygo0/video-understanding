"""Chunk: split a merged timeline into LLM-sized chunks, on segment boundaries."""
from src.merge import format_timeline

DEFAULT_MAX_TOKENS = 1800
CHARS_PER_TOKEN = 4  # rough heuristic, avoids pulling in a full tokenizer just for sizing


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def chunk_events(events: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS) -> list[list[dict]]:
    """Group timeline events into chunks that stay under max_tokens, splitting only
    between events (never mid-sentence/mid-segment)."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0

    for event in events:
        event_tokens = estimate_tokens(event["text"])
        if current and current_tokens + event_tokens > max_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(event)
        current_tokens += event_tokens

    if current:
        chunks.append(current)
    return chunks


def chunk_to_text(events: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS) -> list[str]:
    """Chunk events and render each chunk as timestamped text, ready for summarization."""
    return [format_timeline(chunk) for chunk in chunk_events(events, max_tokens)]
