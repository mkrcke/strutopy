import numpy as np


def test_corpus_length(toy_corpus):
    assert len(toy_corpus.documents) == 50


def test_theta_shape(toy_corpus):
    assert toy_corpus.theta.shape == (50, 3)


def test_theta_rows_sum_to_one(toy_corpus):
    row_sums = toy_corpus.theta.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_beta_shape(toy_corpus):
    assert toy_corpus.beta.shape[0] == 3  # K topics
    assert toy_corpus.beta.shape[1] == 200  # V vocab


def test_documents_are_bow(toy_corpus):
    for doc in toy_corpus.documents:
        assert isinstance(doc, list)
        for item in doc:
            assert isinstance(item, tuple) and len(item) == 2
            assert isinstance(item[0], (int, np.integer))
            assert isinstance(item[1], (int, np.integer))


def test_train_test_split_sizes(toy_corpus):
    assert len(toy_corpus.train_docs) == 40
    assert len(toy_corpus.test_docs) == 10
