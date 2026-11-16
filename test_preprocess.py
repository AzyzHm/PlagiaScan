import pytest

from preprocess import (
    build_fingerprints,
    fingerprint,
    hash_shingle,
    load_corpus,
    make_shingles,
    normalize,
    tokenize,
)


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize("Hello, World!  It's well-known.") == "hello world its well known"


def test_tokenize_basic():
    assert tokenize("The cat, the DOG.") == ["the", "cat", "the", "dog"]


def test_tokenize_stopwords_removed_only_on_request():
    text = "the cat and the dog"
    assert tokenize(text) == ["the", "cat", "and", "the", "dog"]
    assert tokenize(text, remove_stopwords=True) == ["cat", "dog"]


def test_make_shingles_count_and_content():
    tokens = ["a", "b", "c", "d", "e", "f"]
    assert make_shingles(tokens, k=5) == ["a b c d e", "b c d e f"]


def test_make_shingles_short_and_empty_documents():
    assert make_shingles(["a", "b"], k=5) == ["a b"]
    assert make_shingles([], k=5) == []


def test_make_shingles_rejects_bad_k():
    with pytest.raises(ValueError):
        make_shingles(["a"], k=0)


def test_hash_is_deterministic_and_64_bit():
    h = hash_shingle("the quick brown fox jumps")
    assert h == hash_shingle("the quick brown fox jumps")
    assert 0 <= h < 2**64
    assert h != hash_shingle("the quick brown fox jumped")


def test_fingerprint_ignores_case_and_punctuation():
    a = fingerprint("The quick brown fox jumps over the lazy dog.")
    b = fingerprint("the QUICK, brown fox -- jumps over the lazy dog!!!")
    assert a == b


def test_fingerprint_empty_text():
    assert fingerprint("") == set()
    assert fingerprint("... !!!") == set()


def test_load_corpus_and_build_fingerprints(tmp_path):
    (tmp_path / "b.txt").write_text("one two three four five six", encoding="utf-8")
    (tmp_path / "a.txt").write_text("alpha beta gamma delta epsilon", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("not a txt file", encoding="utf-8")

    corpus = load_corpus(tmp_path)
    assert list(corpus) == ["a.txt", "b.txt"]

    prints = build_fingerprints(corpus, k=5)
    assert len(prints["a.txt"]) == 1
    assert len(prints["b.txt"]) == 2


def test_load_corpus_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "missing")
    with pytest.raises(ValueError):
        load_corpus(tmp_path)  # exists but has no .txt files