"""Text preprocessing: normalization, word n-gram shingling and hashing.

Pipeline for one document:
    raw text -> tokens -> shingles (k consecutive words) -> hashed fingerprint set

The fingerprint is a ``set`` of 64-bit integers. Two documents sharing many
fingerprints share many k-word sequences, which is the signal Jaccard
similarity measures.

Note: hashing uses ``hashlib.blake2b`` and NOT Python's built-in ``hash()``.
The built-in is randomized per process, so two workers would disagree on the
value of the same shingle. blake2b is deterministic everywhere, which matters
once fingerprints are shipped between processes and machines.
"""

import hashlib
import re
from pathlib import Path

DEFAULT_K = 5

# Small built-in list, only used when remove_stopwords=True.
STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from as is are was were
    be been being it its this that these those i you he she we they them his
    her their our your not no
    """.split()
)

_APOSTROPHES = re.compile(r"['’]")
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, drop apostrophes, turn all other punctuation into spaces."""
    text = text.lower()
    text = _APOSTROPHES.sub("", text)  # "don't" -> "dont"
    text = _NON_WORD.sub(" ", text)  # "well-known," -> "well known "
    return text.strip()


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    """Normalize ``text`` and split it into a list of words."""
    tokens = normalize(text).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def make_shingles(tokens: list[str], k: int = DEFAULT_K) -> list[str]:
    """Return every run of ``k`` consecutive tokens joined by a space.

    A document shorter than ``k`` tokens yields one shingle (all its tokens),
    so very short documents can still be compared. An empty document yields
    no shingles.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not tokens:
        return []
    if len(tokens) < k:
        return [" ".join(tokens)]
    return [" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]


def hash_shingle(shingle: str) -> int:
    """Deterministic 64-bit hash of a shingle string."""
    digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def fingerprint(text: str, k: int = DEFAULT_K, remove_stopwords: bool = False) -> set[int]:
    """Full pipeline: text -> set of hashed k-word shingles."""
    tokens = tokenize(text, remove_stopwords=remove_stopwords)
    return {hash_shingle(s) for s in make_shingles(tokens, k)}


def load_corpus(directory: str | Path) -> dict[str, str]:
    """Read every ``.txt`` file in ``directory``.

    Returns ``{file name: text}`` ordered by file name.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {directory}")
    corpus = {
        path.name: path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(directory.glob("*.txt"))
    }
    if not corpus:
        raise ValueError(f"No .txt files found in {directory}")
    return corpus


def build_fingerprints(
    corpus: dict[str, str], k: int = DEFAULT_K, remove_stopwords: bool = False
) -> dict[str, set[int]]:
    """Fingerprint every document of a corpus: ``{name: set of hashes}``."""
    return {
        name: fingerprint(text, k=k, remove_stopwords=remove_stopwords)
        for name, text in corpus.items()
    }