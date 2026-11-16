from similarity import jaccard


def test_identical_sets():
    assert jaccard({1, 2, 3}, {1, 2, 3}) == 1.0


def test_disjoint_sets():
    assert jaccard({1, 2}, {3, 4}) == 0.0


def test_partial_overlap():
    # intersection {2, 3} = 2, union {1, 2, 3, 4} = 4
    assert jaccard({1, 2, 3}, {2, 3, 4}) == 0.5


def test_symmetry():
    a, b = {1, 2, 3, 4}, {3, 4, 5}
    assert jaccard(a, b) == jaccard(b, a)


def test_empty_sets_score_zero():
    assert jaccard(set(), set()) == 0.0
    assert jaccard({1}, set()) == 0.0
    assert jaccard(set(), {1}) == 0.0


def test_subset():
    # {1, 2} is contained in {1, 2, 3, 4}: 2 / 4
    assert jaccard({1, 2}, {1, 2, 3, 4}) == 0.5