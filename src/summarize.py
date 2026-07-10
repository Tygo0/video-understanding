"""Map-reduce summarization via a local Ollama model."""
import ollama

DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"

CHUNK_PROMPT = """You are summarizing part of a video's transcript (with any on-screen \
text/visual notes marked "(on-screen)"). Write a concise summary in 100-150 words of what \
happens in this segment. Do not mention timestamps or that this is a transcript excerpt \
— just describe the content.

Segment:
{chunk_text}

Summary:"""

FINAL_PROMPT = """Below are summaries of consecutive segments of a video, in order. Write a \
short, coherent summary (150-250 words) of what the video as a whole is about, synthesizing \
these segments into one narrative. Do not refer to "segments" or "summaries" — describe the \
video's content directly.

Segment summaries:
{combined_summaries}

Overall summary:"""


def _generate(prompt: str, model: str) -> str:
    response = ollama.generate(model=model, prompt=prompt)
    return response["response"].strip()


def summarize_chunk(chunk_text: str, model: str = DEFAULT_MODEL) -> str:
    return _generate(CHUNK_PROMPT.format(chunk_text=chunk_text), model=model)


def summarize_final(chunk_summaries: list[str], model: str = DEFAULT_MODEL) -> str:
    combined = "\n\n".join(f"{i+1}. {s}" for i, s in enumerate(chunk_summaries))
    return _generate(FINAL_PROMPT.format(combined_summaries=combined), model=model)


def summarize_chunks(chunk_texts: list[str], model: str = DEFAULT_MODEL) -> dict:
    """Map-reduce: summarize each chunk, then combine into one final summary."""
    chunk_summaries = [summarize_chunk(c, model=model) for c in chunk_texts]
    final_summary = (
        chunk_summaries[0] if len(chunk_summaries) == 1
        else summarize_final(chunk_summaries, model=model)
    )
    return {"chunk_summaries": chunk_summaries, "final_summary": final_summary}
