import numpy as np
import pytest

from modules.generate_docs import CorpusCreation


@pytest.fixture(scope="session")
def toy_corpus_result():
    """Create a deterministic toy corpus for testing."""
    # Seed legacy RNG (used by init_gamma, init_eta)
    np.random.seed(42)

    # Pre-generate gamma to avoid legacy RNG path in init_gamma
    level = 1
    K = 3
    gamma = np.random.multivariate_normal(
        np.random.standard_normal(level),
        np.diag(np.full(level, 0.001)),
        K - 1,
    )

    corpus = CorpusCreation(
        n_topics=K,
        n_docs=50,
        n_words=50,
        V=200,
        level=level,
        dgp="STM",
        gamma=gamma,
    )
    corpus.generate_documents(remove_terms=True)
    corpus.split_corpus(proportion=0.8)
    return corpus


@pytest.fixture(scope="session")
def toy_corpus(toy_corpus_result):
    return toy_corpus_result


@pytest.fixture(scope="session")
def toy_dictionary(toy_corpus_result):
    return toy_corpus_result.dictionary


@pytest.fixture(scope="session")
def toy_metadata(toy_corpus_result):
    return toy_corpus_result.metadata
