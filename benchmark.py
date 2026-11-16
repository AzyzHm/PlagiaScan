"""Local benchmark: sequential baseline vs multiprocessing.Pool.

T1 is the sequential baseline (baseline.compare_all). For each pool size n:
    speedup    = T1 / Tn
    efficiency = speedup / n

Two timings per pool size:
    cold: pool creation + comparison (what the one-shot CLI pays)
    warm: comparison only, pool already running (what a long-lived TCP worker
          pays per batch once it is up)

Usage:
    python generate_corpus.py bench_corpus --n-docs 300 --n-plagiarized 30
    python benchmark.py bench_corpus --processes 1,2,4,8
"""

import argparse
import os
import time

from baseline import compare_all
from parallel import PairScorer, compare_all_parallel
from preprocess import DEFAULT_K, build_fingerprints, load_corpus


def best_of(repeats: int, fn) -> tuple[float, object]:
    """Run ``fn`` ``repeats`` times; return (fastest time, last result)."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Local speedup benchmark.")
    parser.add_argument("corpus", help="directory containing .txt files")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="shingle size in words")
    parser.add_argument("--threshold", type=float, default=0.05, help="minimum score to keep")
    parser.add_argument("--processes", default="1,2,4,8", help="comma-separated pool sizes")
    parser.add_argument("--repeats", type=int, default=3, help="runs per measurement (best kept)")
    parser.add_argument("--start-method", default=None, help="fork / spawn / forkserver")
    args = parser.parse_args()

    sizes = [int(x) for x in args.processes.split(",")]
    corpus = load_corpus(args.corpus)
    fingerprints = build_fingerprints(corpus, k=args.k)
    n = len(corpus)

    print(f"{n} documents, {n * (n - 1) // 2} pairs, k={args.k}")
    print(f"{os.cpu_count()} logical CPU(s) on this machine, best of {args.repeats} runs\n")

    t1, expected = best_of(args.repeats, lambda: compare_all(fingerprints, args.threshold))
    print(f"T1 (sequential baseline): {t1:.3f}s\n")

    header = f"{'procs':>5} | {'cold (s)':>9} {'speedup':>8} {'eff':>6} | {'warm (s)':>9} {'speedup':>8} {'eff':>6}"
    print(header)
    print("-" * len(header))

    for p in sizes:
        t_cold, res_cold = best_of(
            args.repeats,
            lambda: compare_all_parallel(
                fingerprints, args.threshold, processes=p, start_method=args.start_method
            ),
        )
        with PairScorer(fingerprints, processes=p, start_method=args.start_method) as scorer:
            compare_all_parallel(fingerprints, args.threshold, scorer=scorer)  # warm-up
            t_warm, res_warm = best_of(
                args.repeats,
                lambda: compare_all_parallel(fingerprints, args.threshold, scorer=scorer),
            )
        if res_cold != expected or res_warm != expected:
            raise SystemExit(f"results differ from baseline with {p} process(es)!")
        print(
            f"{p:>5} | {t_cold:>9.3f} {t1 / t_cold:>7.2f}x {t1 / t_cold / p:>6.0%}"
            f" | {t_warm:>9.3f} {t1 / t_warm:>7.2f}x {t1 / t_warm / p:>6.0%}"
        )

    print("\nAll parallel results identical to the baseline.")


if __name__ == "__main__":
    main()