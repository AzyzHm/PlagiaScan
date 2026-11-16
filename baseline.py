"""Sequential baseline: compare every pair of documents in a single process.

This is the reference implementation. Later phases (multiprocessing, then
TCP workers) must produce exactly the same ranked results, and the time it
takes here is T1 for the speedup analysis.

Usage:
    python baseline.py corpus --threshold 0.05 --top 15
"""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

from preprocess import DEFAULT_K, build_fingerprints, load_corpus
from similarity import jaccard

Result = tuple[str, str, float]


def compare_all(fingerprints: dict[str, set[int]], threshold: float = 0.0) -> list[Result]:
    """Compare all N(N-1)/2 pairs; keep those scoring >= ``threshold``.

    Returns ``(doc_a, doc_b, score)`` tuples, highest score first. Ties are
    broken by document names so the output order is fully deterministic.
    """
    names = sorted(fingerprints)
    results: list[Result] = []
    for a, b in combinations(names, 2):
        score = jaccard(fingerprints[a], fingerprints[b])
        if score >= threshold:
            results.append((a, b, score))
    results.sort(key=lambda r: (-r[2], r[0], r[1]))
    return results


def load_truth(path: str | Path) -> set[frozenset[str]]:
    """Load the ground-truth plagiarized pairs written by generate_corpus.py."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {frozenset((p["source"], p["target"])) for p in data["pairs"]}


def evaluate(
    results: list[Result], truth: set[frozenset[str]], threshold: float
) -> tuple[float, float]:
    """Precision and recall of the pairs scoring >= ``threshold``."""
    flagged = {frozenset((a, b)) for a, b, s in results if s >= threshold}
    true_positives = len(flagged & truth)
    precision = true_positives / len(flagged) if flagged else 0.0
    recall = true_positives / len(truth) if truth else 0.0
    return precision, recall


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential plagiarism baseline.")
    parser.add_argument("corpus", help="directory containing .txt files")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="shingle size in words")
    parser.add_argument("--threshold", type=float, default=0.05, help="minimum score to report")
    parser.add_argument("--top", type=int, default=15, help="how many pairs to print")
    parser.add_argument("--stopwords", action="store_true", help="remove stopwords")
    parser.add_argument("--truth", help="ground_truth.json (default: <corpus>/ground_truth.json)")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)

    t0 = time.perf_counter()
    fingerprints = build_fingerprints(corpus, k=args.k, remove_stopwords=args.stopwords)
    t_prep = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = compare_all(fingerprints, threshold=args.threshold)
    t_compare = time.perf_counter() - t0

    n = len(corpus)
    print(f"{n} documents, {n * (n - 1) // 2} pairs, k={args.k}")
    print(f"preprocessing: {t_prep:.3f}s | comparison: {t_compare:.3f}s")
    print(f"{len(results)} pair(s) with score >= {args.threshold}\n")

    for a, b, score in results[: args.top]:
        print(f"{score:.4f}  {a}  <->  {b}")

    truth_path = Path(args.truth) if args.truth else Path(args.corpus) / "ground_truth.json"
    if truth_path.is_file():
        precision, recall = evaluate(results, load_truth(truth_path), args.threshold)
        print(f"\nvs ground truth: precision={precision:.2f} recall={recall:.2f}")


if __name__ == "__main__":
    main()