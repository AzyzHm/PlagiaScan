"""Similarity metrics between document fingerprints."""


def jaccard(a: set[int], b: set[int]) -> float:
    """Jaccard similarity |A ∩ B| / |A ∪ B|, between 0.0 and 1.0.

    Two empty sets return 0.0: an empty document is not "similar" to anything,
    and this avoids a division by zero.
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union