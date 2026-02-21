import numpy as np

from modules.heldout import cut_in_half, eval_heldout


def test_eval_heldout_finite_negative():
    """eval_heldout should return a finite negative scalar for valid inputs."""
    K, V, N = 3, 20, 5
    rng = np.random.default_rng(99)

    beta = rng.dirichlet(np.ones(V), size=K)
    theta = rng.dirichlet(np.ones(K), size=N)

    # Create synthetic BoW docs
    docs = []
    for i in range(N):
        words = rng.choice(V, size=10)
        unique, counts = np.unique(words, return_counts=True)
        docs.append(list(zip(unique, counts)))

    result = eval_heldout(docs, theta, beta)
    assert np.isfinite(result)
    assert result < 0


def test_cut_in_half_splits_correctly():
    """cut_in_half should produce even/odd index splits."""
    docs = [
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        [(4, 1), (5, 2), (6, 3)],
    ]
    first, second = cut_in_half(docs)

    # First half: indices 0, 2 of each doc
    assert len(first) == 2
    assert list(first[0]) == [(0, 1), (2, 3)]
    assert list(first[1]) == [(4, 1), (6, 3)]

    # Second half: indices 1, 3 of each doc
    assert list(second[0]) == [(1, 2), (3, 4)]
    assert list(second[1]) == [(5, 2)]
