"""Synthetic corpus generator with injected plagiarism and ground truth.

Creates ``n_docs`` documents of pseudo-random words. Then, for a number of
disjoint (source, target) document pairs, a contiguous passage of the source
is copied into the target at a random position, with a small fraction of its
words randomly replaced ("small edits"). The injected pairs are saved in
``ground_truth.json`` so detection accuracy can be measured.

Usage:
    python generate_corpus.py corpus --n-docs 100 --n-plagiarized 10 --seed 42
"""

import argparse
import json
import random
from pathlib import Path

CONSONANTS = "bcdfghjklmnprstvz"
VOWELS = "aeiou"


def make_vocab(rng: random.Random, size: int) -> list[str]:
    """Build ``size`` distinct pronounceable pseudo-words (deterministic)."""
    words: set[str] = set()
    while len(words) < size:
        n_syllables = rng.choice((2, 3, 3, 4))
        words.add(
            "".join(rng.choice(CONSONANTS) + rng.choice(VOWELS) for _ in range(n_syllables))
        )
    return sorted(words)  # sorted: set order is not stable across runs


def random_words(rng: random.Random, vocab: list[str], n: int) -> list[str]:
    return [rng.choice(vocab) for _ in range(n)]


def inject_passage(
    rng: random.Random,
    vocab: list[str],
    src_words: list[str],
    dst_words: list[str],
    fraction: float,
    edit_rate: float,
) -> tuple[list[str], int]:
    """Copy a passage of ``src_words`` into ``dst_words``.

    ``fraction``: passage length as a share of the source length.
    ``edit_rate``: probability that each copied word is replaced by a random one.
    Returns the new target words and the passage length in words.
    """
    length = max(1, int(len(src_words) * fraction))
    start = rng.randrange(len(src_words) - length + 1)
    passage = [
        rng.choice(vocab) if rng.random() < edit_rate else w
        for w in src_words[start : start + length]
    ]
    pos = rng.randrange(len(dst_words) + 1)
    return dst_words[:pos] + passage + dst_words[pos:], length


def format_text(rng: random.Random, words: list[str]) -> str:
    """Lay words out as capitalized sentences and paragraphs.

    Sentence boundaries are random, so a copied passage looks different in the
    source and in the target. Normalization must see through that.
    """
    sentences = []
    i = 0
    while i < len(words):
        n = rng.randint(8, 16)
        sentences.append(" ".join(words[i : i + n]).capitalize() + ".")
        i += n
    paragraphs = [" ".join(sentences[j : j + 5]) for j in range(0, len(sentences), 5)]
    return "\n\n".join(paragraphs) + "\n"


def generate(
    out_dir: str | Path,
    n_docs: int = 100,
    n_plagiarized: int = 10,
    min_words: int = 800,
    max_words: int = 2000,
    vocab_size: int = 5000,
    seed: int = 42,
) -> list[dict]:
    """Write the corpus and ``ground_truth.json``; return the injected pairs."""
    if 2 * n_plagiarized > n_docs:
        raise ValueError("need n_docs >= 2 * n_plagiarized (pairs use disjoint documents)")
    if min_words > max_words:
        raise ValueError("min_words must be <= max_words")

    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("doc_*.txt"):  # avoid mixing with a previous corpus
        old.unlink()

    vocab = make_vocab(rng, vocab_size)
    width = max(3, len(str(n_docs - 1)))
    names = [f"doc_{i:0{width}d}.txt" for i in range(n_docs)]
    docs = {name: random_words(rng, vocab, rng.randint(min_words, max_words)) for name in names}

    pairs: list[dict] = []
    chosen = rng.sample(names, 2 * n_plagiarized)
    for source, target in zip(chosen[:n_plagiarized], chosen[n_plagiarized:]):
        fraction = round(rng.uniform(0.2, 0.6), 3)
        edit_rate = round(rng.uniform(0.0, 0.1), 3)
        docs[target], length = inject_passage(
            rng, vocab, docs[source], docs[target], fraction, edit_rate
        )
        pairs.append(
            {
                "source": source,
                "target": target,
                "passage_words": length,
                "fraction": fraction,
                "edit_rate": edit_rate,
            }
        )

    for name, words in docs.items():
        (out_dir / name).write_text(format_text(rng, words), encoding="utf-8")

    truth = {"seed": seed, "n_docs": n_docs, "pairs": pairs}
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic plagiarism corpus.")
    parser.add_argument("out_dir", help="output directory")
    parser.add_argument("--n-docs", type=int, default=100)
    parser.add_argument("--n-plagiarized", type=int, default=10, help="number of injected pairs")
    parser.add_argument("--min-words", type=int, default=800)
    parser.add_argument("--max-words", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = generate(
        args.out_dir,
        n_docs=args.n_docs,
        n_plagiarized=args.n_plagiarized,
        min_words=args.min_words,
        max_words=args.max_words,
        seed=args.seed,
    )
    print(f"Wrote {args.n_docs} documents to {args.out_dir} with {len(pairs)} injected pairs.")


if __name__ == "__main__":
    main()