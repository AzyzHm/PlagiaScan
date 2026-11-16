import pytest

from baseline import compare_all
from generate_corpus import generate
from parallel import (
    PairScorer,
    compare_all_parallel,
    default_chunksize,
    make_pairs,
)
from preprocess import build_fingerprints, load_corpus


@pytest.fixture(scope="module")
def fingerprints(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    generate(out, n_docs=24, n_plagiarized=4, min_words=300, max_words=600, seed=3)
    return build_fingerprints(load_corpus(out))


@pytest.mark.parametrize("processes", [1, 2, 4])
def test_matches_baseline_exactly(fingerprints, processes):
    expected = compare_all(fingerprints, threshold=0.0)
    assert compare_all_parallel(fingerprints, 0.0, processes=processes) == expected


def test_matches_baseline_with_threshold(fingerprints):
    expected = compare_all(fingerprints, threshold=0.05)
    assert compare_all_parallel(fingerprints, 0.05, processes=2) == expected


@pytest.mark.parametrize("chunksize", [1, 7, 10_000])
def test_result_does_not_depend_on_chunksize(fingerprints, chunksize):
    expected = compare_all(fingerprints, threshold=0.0)
    assert compare_all_parallel(fingerprints, 0.0, processes=2, chunksize=chunksize) == expected


def test_works_with_spawn_start_method(fingerprints):
    """Windows and macOS use 'spawn': globals are not inherited, only initargs."""
    expected = compare_all(fingerprints, threshold=0.05)
    assert compare_all_parallel(fingerprints, 0.05, processes=2, start_method="spawn") == expected


def test_scorer_pool_is_reusable_across_calls(fingerprints):
    pairs = make_pairs(fingerprints)
    with PairScorer(fingerprints, processes=2) as scorer:
        first = scorer.score(pairs[:50])
        second = scorer.score(pairs[50:120], chunksize=5)
    assert [(a, b) for a, b, _ in first] == pairs[:50]
    assert [(a, b) for a, b, _ in second] == pairs[50:120]


def test_make_pairs_covers_every_pair_once(fingerprints):
    n = len(fingerprints)
    pairs = make_pairs(fingerprints)
    assert len(pairs) == n * (n - 1) // 2
    assert len(set(pairs)) == len(pairs)
    assert all(a < b for a, b in pairs)


def test_fewer_than_two_documents():
    assert compare_all_parallel({}, processes=2) == []
    assert compare_all_parallel({"only.txt": {1, 2, 3}}, processes=2) == []


def test_invalid_arguments(fingerprints):
    with pytest.raises(ValueError):
        PairScorer(fingerprints, processes=0)
    with PairScorer(fingerprints, processes=1) as scorer:
        with pytest.raises(ValueError):
            scorer.score([("a", "b")], chunksize=0)


def test_default_chunksize():
    assert default_chunksize(0, 4) == 1
    assert default_chunksize(4950, 4) == 310  # ceil(4950 / 16)
    assert default_chunksize(10, 8) == 1