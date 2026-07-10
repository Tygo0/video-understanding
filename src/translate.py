"""Translation: convert the final summary into a target language via NLLB-200.

A dedicated translation model, not the summarization LLM: qwen2.5:7b-instruct-q4_K_M
proved unreliable at non-English output (see src/summarize.py), including outright
mistranslations when asked to translate directly. NLLB-200 is purpose-built for
translation and covers 200 languages with one small (~600M param) model.
"""
import re
from pathlib import Path

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Simple language codes (as used in --target-language) -> NLLB's FLORES-200 codes.
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "tr": "tur_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
}


class Translator:
    """Lazily-initialized NLLB-200 model + tokenizer."""

    _model = None
    _tokenizer = None

    @classmethod
    def get(cls):
        if cls._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            cls._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            cls._model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            # The base generation config ships a max_length that otherwise triggers a
            # harmless-but-noisy warning every call since we pass max_new_tokens instead.
            cls._model.generation_config.max_length = None
        return cls._model, cls._tokenizer


def _translate_sentence(sentence: str, target_token_id: int) -> str:
    model, tokenizer = Translator.get()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    generated = model.generate(
        **inputs, forced_bos_token_id=target_token_id, max_new_tokens=256, num_beams=4
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def translate(text: str, target_language: str, source_language: str = "en") -> str:
    """Translate text from source_language to target_language (simple codes, e.g. 'tr').

    Translates sentence-by-sentence rather than the whole text in one pass: NLLB-200
    was observed to silently truncate multi-sentence paragraphs (stopping partway
    through with no error), producing an incomplete translation. Per-sentence
    translation avoids this and also improved word-choice accuracy in testing.
    """
    if target_language not in NLLB_LANGUAGE_CODES:
        raise ValueError(
            f"Unsupported target language: {target_language!r}. "
            f"Supported: {sorted(NLLB_LANGUAGE_CODES)}"
        )
    if source_language not in NLLB_LANGUAGE_CODES:
        raise ValueError(f"Unsupported source language: {source_language!r}")

    model, tokenizer = Translator.get()
    tokenizer.src_lang = NLLB_LANGUAGE_CODES[source_language]
    target_token_id = tokenizer.convert_tokens_to_ids(NLLB_LANGUAGE_CODES[target_language])

    paragraphs = text.strip().split("\n\n")
    translated_paragraphs = []
    for paragraph in paragraphs:
        sentences = SENTENCE_SPLIT_RE.split(paragraph.strip())
        translated_sentences = [_translate_sentence(s, target_token_id) for s in sentences if s]
        translated_paragraphs.append(" ".join(translated_sentences))
    return "\n\n".join(translated_paragraphs)


def translate_to_file(
    text: str, target_language: str, out_path: Path, source_language: str = "en"
) -> Path:
    translated = translate(text, target_language, source_language=source_language)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(translated)
    return out_path
