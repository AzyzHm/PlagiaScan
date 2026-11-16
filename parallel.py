"""Single-machine parallel comparison with multiprocessing.Pool.

Same job as baseline.compare_all, spread over several CPU cores. The output
must be identical to the baseline (same pairs, same scores, same order).

Design notes
------------
* Fingerprints are shipped to each pool process ONCE, through the Pool
  initializer, and kept in a module-level global. Tasks are then just pairs of
  document names (a few bytes each). Sending the sets with every task would
  make pickling cost more than the comparison itself.
* ``Pool.starmap(..., chunksize=N)`` groups N pairs per task: bigger chunks
  mean less inter-process traffic, smaller chunks balance load better.
* ``PairScorer`` keeps its pool alive between calls. A TCP worker (Phase 3)
  receives many batches and must not pay the pool start-up cost for each one.
* Everything works with the "spawn" start method (default on Windows and
  macOS), so the entry point must be guarded by ``if __name__ == "__main__"``.

Usage:
    python parallel.py corpus --processes 4 --threshold 0.05 --verify
"""

import argparse
import math
import multiprocessing as mp
import os
import time
from itertools import combinations

from baseline import Result, compare_all
from preprocess import DEFAULT_K, build_fingerprints, load_corpus
from similarity import jaccard

# Set once in every pool process by _init_worker (see module docstring).
_FINGERPRINTS: dict[str, set[int]] = {}

CHUNKS_PER_PROCESS = 4  # default chunksize aims for ~4 chunks per process


def _init_worker(fingerprints: dict[str, set[int]]) -> None:
    """Pool initializer: runs once in each worker process."""
    global _FINGERPRINTS
    _FINGERPRINTS = fingerprints


def _score_pair(a: str, b: str) -> Result:
    """Task run inside a pool process: Jaccard score of one pair."""
    return a, b, jaccard(_FINGERPRINTS[a], _FINGERPRINTS[b])


def default_chunksize(n_pairs: int, processes: int) -> int:
    """About ``CHUNKS_PER_PROCESS`` chunks per process, at least 1 pair each."""
    return max(1, math.ceil(n_pairs / (processes * CHUNKS_PER_PROCESS)))


class PairScorer:
    """A pool of processes that already holds the fingerprints.

    Use it as a context manager::

        with PairScorer(fingerprints, processes=4) as scorer:
            scored = scorer.score(pairs)
    """

    def __init__(
        self,
        fingerprints: dict[str, set[int]],
        processes: int | None = None,
        start_method: str | None = None,
    ) -> None:
        self.processes = processes if processes is not None else (os.cpu_count() or 1)
        if self.processes < 1:
            raise ValueError("processes must be >= 1")
        ctx = mp.get_context(start_method)  # None = platform default
        self._pool = ctx.Pool(
            self.processes, initializer=_init_worker, initargs=(fingerprints,)
        )

    def score(self, pairs: list[tuple[str, str]], chunksize: int | None = None) -> list[Result]:
        """Score every pair; results come back in the same order as ``pairs``."""
        pairs = list(pairs)
        if not pairs:
            return []
        if chunksize is None:
            chunksize = default_chunksize(len(pairs), self.processes)
        elif chunksize < 1:
            raise ValueError("chunksize must be >= 1")
        return self._pool.starmap(_score_pair, pairs, chunksize)

    def close(self) -> None:
        self._pool.close()
        self._pool.join()

    def __enter__(self) -> "PairScorer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self._pool.terminate()
            self._pool.join()


def make_pairs(fingerprints: dict[str, set[int]]) -> list[tuple[str, str]]:
    """All N(N-1)/2 pairs of document names, in the baseline's order."""
    return list(combinations(sorted(fingerprints), 2))


def compare_all_parallel(
    fingerprints: dict[str, set[int]],
    threshold: float = 0.0,
    processes: int | None = None,
    chunksize: int | None = None,
    start_method: str | None = None,
    scorer: PairScorer | None = None,
) -> list[Result]:
    """Parallel drop-in replacement for ``baseline.compare_all``.

    Pass an existing ``scorer`` to reuse a warm pool (``processes`` and
    ``start_method`` are then ignored, and the caller closes the pool).
    """
    pairs = make_pairs(fingerprints)
    if not pairs:
        return []
    if scorer is not None:
        scored = scorer.score(pairs, chunksize)
    else:
        with PairScorer(fingerprints, processes, start_method) as own_scorer:
            scored = own_scorer.score(pairs, chunksize)
    results = [r for r in scored if r[2] >= threshold]
    # Same sort key as baseline.compare_all, so the output is identical.
    results.sort(key=lambda r: (-r[2], r[0], r[1]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel plagiarism comparison (one machine).")
    parser.add_argument("corpus", help="directory containing .txt files")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="shingle size in words")
    parser.add_argument("--threshold", type=float, default=0.05, help="minimum score to report")
    parser.add_argument("--top", type=int, default=15, help="how many pairs to print")
    parser.add_argument("--stopwords", action="store_true", help="remove stopwords")
    parser.add_argument("--processes", type=int, default=None, help="pool size (default: CPU count)")
    parser.add_argument("--chunksize", type=int, default=None, help="pairs per task (default: auto)")
    parser.add_argument("--verify", action="store_true", help="also run the baseline and compare")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    fingerprints = build_fingerprints(corpus, k=args.k, remove_stopwords=args.stopwords)

    t0 = time.perf_counter()
    results = compare_all_parallel(
        fingerprints, args.threshold, processes=args.processes, chunksize=args.chunksize
    )
    t_par = time.perf_counter() - t0

    n = len(corpus)
    procs = args.processes or os.cpu_count() or 1
    print(f"{n} documents, {n * (n - 1) // 2} pairs, k={args.k}, {procs} process(es)")
    print(f"parallel comparison (incl. pool start-up): {t_par:.3f}s")
    print(f"{len(results)} pair(s) with score >= {args.threshold}\n")
    for a, b, score in results[: args.top]:
        print(f"{score:.4f}  {a}  <->  {b}")

    if args.verify:
        t0 = time.perf_counter()
        expected = compare_all(fingerprints, args.threshold)
        t_seq = time.perf_counter() - t0
        status = "IDENTICAL" if results == expected else "MISMATCH"
        print(f"\nbaseline: {t_seq:.3f}s -> results {status}")
        if results != expected:
            raise SystemExit(1)


if __name__ == "__main__":
    main()