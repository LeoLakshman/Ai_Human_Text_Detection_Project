"""
Linguistic / stylistic feature extraction for AI vs Human text detection.

Implemented with pure Python + regex (no nltk/textstat dependency required),
so this module works identically in the notebook, the Streamlit app, and any
environment with just the standard library + numpy.

Features extracted (per document):
- sentence_count, avg_sentence_length (words/sentence)
- avg_word_length
- type_token_ratio        (vocabulary richness: unique words / total words)
- hapax_legomena_ratio    (words used exactly once / total words)
- punctuation_density     (punctuation chars / total chars)
- comma_density, period_density
- avg_word_length_std     (variability in word length)
- flesch_reading_ease     (readability, computed with a syllable heuristic)
- flesch_kincaid_grade
- stopword_ratio          (ratio of common function words)
- contraction_ratio       (use of contractions, e.g. don't, it's)
- avg_syllables_per_word
- long_word_ratio         (words with 7+ letters)
- passive_voice_hits      (rough heuristic count of passive constructions)
"""

import re
import numpy as np

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_SPLIT_RE = re.compile(r"[.!?]+")
_VOWEL_RE = re.compile(r"[aeiouyAEIOUY]+")
_PUNCT_RE = re.compile(r"[^\w\s]")

# A compact stopword list (no nltk dependency)
_STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who
who's whom why why's with won't would wouldn't you you'd you'll you're you've
your yours yourself yourselves
""".split())

_CONTRACTIONS = re.compile(
    r"\b\w+'(t|s|re|ve|ll|d|m)\b", re.IGNORECASE
)
_PASSIVE_RE = re.compile(
    r"\b(is|are|was|were|be|been|being)\s+\w+ed\b", re.IGNORECASE
)


def _count_syllables(word: str) -> int:
    word = word.lower()
    groups = _VOWEL_RE.findall(word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def extract_linguistic_features(text: str) -> dict:
    """Return a dict of linguistic/stylistic features for a single document."""
    text = text if isinstance(text, str) else ""
    words = _WORD_RE.findall(text)
    n_words = len(words)
    sentences = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    n_sent = max(len(sentences), 1)

    if n_words == 0:
        return {k: 0.0 for k in [
            "sentence_count", "avg_sentence_length", "avg_word_length",
            "type_token_ratio", "hapax_legomena_ratio", "punctuation_density",
            "comma_density", "period_density", "word_length_std",
            "flesch_reading_ease", "flesch_kincaid_grade", "stopword_ratio",
            "contraction_ratio", "avg_syllables_per_word", "long_word_ratio",
            "passive_voice_hits",
        ]}

    lower_words = [w.lower() for w in words]
    word_lengths = np.array([len(w) for w in words])
    unique_words = set(lower_words)
    from collections import Counter
    counts = Counter(lower_words)
    hapax = sum(1 for w, c in counts.items() if c == 1)

    n_chars = max(len(text), 1)
    n_punct = len(_PUNCT_RE.findall(text))
    n_commas = text.count(",")
    n_periods = text.count(".")

    syllables = [_count_syllables(w) for w in words]
    total_syllables = sum(syllables)
    avg_syll = total_syllables / n_words

    # Flesch Reading Ease / Flesch-Kincaid Grade (standard formulas)
    words_per_sent = n_words / n_sent
    fre = 206.835 - 1.015 * words_per_sent - 84.6 * avg_syll
    fkg = 0.39 * words_per_sent + 11.8 * avg_syll - 15.59

    n_stop = sum(1 for w in lower_words if w in _STOPWORDS)
    n_contractions = len(_CONTRACTIONS.findall(text))
    n_long_words = sum(1 for w in words if len(w) >= 7)
    n_passive = len(_PASSIVE_RE.findall(text))

    return {
        "sentence_count": float(n_sent),
        "avg_sentence_length": float(words_per_sent),
        "avg_word_length": float(word_lengths.mean()),
        "type_token_ratio": float(len(unique_words) / n_words),
        "hapax_legomena_ratio": float(hapax / n_words),
        "punctuation_density": float(n_punct / n_chars),
        "comma_density": float(n_commas / n_sent),
        "period_density": float(n_periods / n_sent),
        "word_length_std": float(word_lengths.std()),
        "flesch_reading_ease": float(fre),
        "flesch_kincaid_grade": float(fkg),
        "stopword_ratio": float(n_stop / n_words),
        "contraction_ratio": float(n_contractions / n_words),
        "avg_syllables_per_word": float(avg_syll),
        "long_word_ratio": float(n_long_words / n_words),
        "passive_voice_hits": float(n_passive / n_sent),
    }


LINGUISTIC_FEATURE_NAMES = [
    "sentence_count", "avg_sentence_length", "avg_word_length",
    "type_token_ratio", "hapax_legomena_ratio", "punctuation_density",
    "comma_density", "period_density", "word_length_std",
    "flesch_reading_ease", "flesch_kincaid_grade", "stopword_ratio",
    "contraction_ratio", "avg_syllables_per_word", "long_word_ratio",
    "passive_voice_hits",
]


def extract_linguistic_matrix(texts):
    """Vectorized helper: list[str] -> np.ndarray of shape (n_docs, n_features)."""
    rows = [extract_linguistic_features(t) for t in texts]
    return np.array([[r[name] for name in LINGUISTIC_FEATURE_NAMES] for r in rows])


def simple_clean_text(text: str) -> str:
    """Light cleaning for TF-IDF / tokenization: lowercase, strip URLs, collapse whitespace."""
    text = text if isinstance(text, str) else ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_tokenize(text: str):
    """Basic word tokenizer (alpha tokens, lowercase) used for embeddings/sequence models."""
    return _WORD_RE.findall(text.lower())
