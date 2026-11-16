import pytest

from baseline import compare_all, evaluate, load_truth
from generate_corpus import generate
from preprocess import build_fingerprints, load_corpus


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    generate(out, n_docs=20, n_plagiarized=3, min_words=300, max_words=600, seed=1)
    return out


@pytest.fixture(scope="module")
def fingerprints(corpus_dir):
    return build_fingerprints(load_corpus(corpus_dir))


def test_compares_every_pair_once(fingerprints):
    n = len(fingerprints)
    results = compare_all(fingerprints, threshold=0.0)
    assert len(results) == n * (n - 1) // 2
    assert len({(a, b) for a, b, _ in results}) == len(results)
    assert all(a < b for a, b, _ in results)


def test_results_sorted_by_descending_score(fingerprints):
    scores = [s for _, _, s in compare_all(fingerprints)]
    assert scores == sorted(scores, reverse=True)


def test_injected_pairs_rank_first(corpus_dir, fingerprints):
    truth = load_truth(corpus_dir / "ground_truth.json")
    results = compare_all(fingerprints)
    top = {frozenset((a, b)) for a, b, _ in results[: len(truth)]}
    assert top == truth


def test_unrelated_pairs_score_zero(corpus_dir, fingerprints):
    truth = load_truth(corpus_dir / "ground_truth.json")
    for a, b, score in compare_all(fingerprints):
        if frozenset((a, b)) not in truth:
            assert score == 0.0


def test_threshold_filters_results(fingerprints):
    results = compare_all(fingerprints, threshold=0.05)
    assert len(results) == 3
    assert all(s >= 0.05 for _, _, s in results)


def test_perfect_precision_and_recall(corpus_dir, fingerprints):
    truth = load_truth(corpus_dir / "ground_truth.json")
    precision, recall = evaluate(compare_all(fingerprints), truth, threshold=0.05)
    assert precision == 1.0
    assert recall == 1.0


def test_generation_is_reproducible(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    pairs_a = generate(a, n_docs=10, n_plagiarized=2, min_words=100, max_words=200, seed=7)
    pairs_b = generate(b, n_docs=10, n_plagiarized=2, min_words=100, max_words=200, seed=7)
    assert pairs_a == pairs_b
    assert (a / "doc_000.txt").read_text() == (b / "doc_000.txt").read_text()


def test_generate_rejects_too_many_pairs(tmp_path):
    with pytest.raises(ValueError):
        generate(tmp_path, n_docs=5, n_plagiarized=3)