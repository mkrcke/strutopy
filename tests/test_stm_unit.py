import numpy as np
from scipy.sparse import issparse

from modules.stm import STM, create_dtm


def test_create_dtm():
    """create_dtm should produce a sparse matrix from BoW input."""
    docs = [
        [(0, 2), (1, 3)],
        [(1, 1), (2, 5)],
    ]
    dtm = create_dtm(docs)
    assert issparse(dtm)
    assert dtm.shape == (2, 3)
    assert dtm[0, 0] == 2
    assert dtm[0, 1] == 3
    assert dtm[1, 1] == 1
    assert dtm[1, 2] == 5


def test_stm_random_init(toy_corpus, toy_dictionary, toy_metadata):
    """STM with random init and CTM model should initialize with correct shapes."""
    np.random.seed(42)
    train_docs = toy_corpus.train_docs
    K = 3
    N_train = len(train_docs)

    model = STM(
        documents=train_docs,
        dictionary=toy_dictionary,
        content=False,
        K=K,
        X=toy_metadata[:N_train],
        kappa_interactions=False,
        max_em_iter=1,
        sigma_prior=0,
        convergence_threshold=1e-5,
        init_type="random",
        model_type="CTM",
    )

    assert model.beta.shape == (K, len(toy_dictionary))
    assert model.theta.shape == (N_train, K)
    assert model.sigma.shape == (K - 1, K - 1)
    assert model.eta.shape == (N_train, K - 1)
