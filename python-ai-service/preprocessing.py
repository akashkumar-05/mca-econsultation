"""
Text Preprocessing Module for MCA eConsultation AI Service.

Implements a robust NLP preprocessing pipeline:
  1. URL removal
  2. Email removal
  3. Emoji removal
  4. Special character normalization
  5. Stopword removal (English + domain-specific)
  6. Whitespace normalization

This module ensures consistent input quality across all downstream
models (sentiment, summarization, word cloud).
"""

import re
import logging
import unicodedata

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# NLTK STOPWORDS (loaded once)
# ─────────────────────────────────────────────────────────────
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.corpus import stopwords as nltk_stopwords
    ENGLISH_STOPWORDS = set(nltk_stopwords.words("english"))
    logger.info(f"NLTK stopwords loaded: {len(ENGLISH_STOPWORDS)} words")
except Exception as e:
    logger.warning(f"NLTK stopwords unavailable, using minimal set: {e}")
    ENGLISH_STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "because", "but", "and", "or",
        "if", "while", "this", "that", "these", "those", "i", "me", "my",
        "we", "our", "you", "your", "he", "him", "his", "she", "her",
        "it", "its", "they", "them", "their", "what", "which", "who",
    }

# Regex patterns compiled once for performance
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
SPECIAL_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9\s.,;:!?'\"-]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str, remove_stopwords: bool = False,
               extra_stopwords: set = None) -> str:
    """
    Clean and normalize a single text string.

    Pipeline:
      1. Unicode normalization (NFKD)
      2. URL removal
      3. Email removal
      4. Emoji removal
      5. Special character removal
      6. Lowercasing
      7. Optional stopword removal
      8. Whitespace normalization

    Args:
        text: Raw input text
        remove_stopwords: Whether to remove English + domain stopwords
        extra_stopwords: Additional stopwords to remove

    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # Remove URLs
    text = URL_PATTERN.sub("", text)

    # Remove emails
    text = EMAIL_PATTERN.sub("", text)

    # Remove emojis
    text = EMOJI_PATTERN.sub("", text)

    # Remove special characters (keep basic punctuation)
    text = SPECIAL_CHARS_PATTERN.sub(" ", text)

    # Lowercase
    text = text.lower()

    # Stopword removal (optional — not used for sentiment analysis)
    if remove_stopwords:
        stop_set = ENGLISH_STOPWORDS.copy()
        if extra_stopwords:
            stop_set.update(extra_stopwords)
        words = text.split()
        words = [w for w in words if w not in stop_set and len(w) > 2]
        text = " ".join(words)

    # Normalize whitespace
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()

    return text


def clean_batch(texts: list, remove_stopwords: bool = False,
                extra_stopwords: set = None) -> list:
    """
    Clean a batch of texts.

    Args:
        texts: List of raw text strings
        remove_stopwords: Whether to remove stopwords
        extra_stopwords: Additional domain stopwords

    Returns:
        List of cleaned text strings
    """
    cleaned = []
    for t in texts:
        c = clean_text(t, remove_stopwords=remove_stopwords,
                       extra_stopwords=extra_stopwords)
        if c:  # skip empty results
            cleaned.append(c)
    return cleaned


def clean_for_sentiment(texts: list) -> list:
    """
    Light cleaning for sentiment analysis.
    Preserves punctuation and sentence structure since RoBERTa
    benefits from natural language patterns.
    """
    return clean_batch(texts, remove_stopwords=False)
