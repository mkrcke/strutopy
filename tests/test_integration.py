import json
import os

import numpy as np
import pytest

from modules.generate_docs import CorpusCreation
from modules.heldout import eval_heldout
from modules.stm import STM

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "baseline_metrics.json")


def _run_toy_pipeline():
    """Run a deterministic toy pipeline and return metrics."""
    np.random.seed(42)

    K, V, N, n_words, level = 3, 200, 50, 50, 1

    gamma = np.random.multivariate_normal(
        np.random.standard_normal(level),
        np.diag(np.full(level, 0.001)),
        K - 1,
    )

    corpus = CorpusCreation(
        n_topics=K,
        n_docs=N,
        n_words=n_words,
        V=V,
        level=level,
        dgp="STM",
        gamma=gamma,
    )
    corpus.generate_documents(remove_terms=True)
    corpus.split_corpus(proportion=0.8)

    train_docs = corpus.train_docs
    N_train = len(train_docs)

    np.random.seed(42)

    model = STM(
        documents=train_docs,
        dictionary=corpus.dictionary,
        content=False,
        K=K,
        X=corpus.metadata[:N_train],
        kappa_interactions=False,
        max_em_iter=2,
        sigma_prior=0,
        convergence_threshold=1e-5,
        init_type="random",
        model_type="CTM",
    )
    model.expectation_maximization(saving=False)

    heldout_ll = eval_heldout(corpus.test_2_docs, model.theta, model.beta)

    return {
        "beta_shape": list(model.beta.shape),
        "theta_shape": list(model.theta.shape),
        "sigma_shape": list(model.sigma.shape),
        "final_bound": float(model.last_bounds[-1]),
        "heldout_ll": float(heldout_ll),
        "theta_row_sums_mean": float(np.mean(model.theta.sum(axis=1))),
        "beta_row_sums_mean": float(np.mean(model.beta.sum(axis=1))),
    }


def test_toy_pipeline_shapes():
    metrics = _run_toy_pipeline()
    K, N_train = 3, 40
    assert metrics["beta_shape"][0] == K
    assert metrics["theta_shape"] == [N_train, K]
    assert metrics["sigma_shape"] == [K - 1, K - 1]


def test_toy_pipeline_probabilities():
    metrics = _run_toy_pipeline()
    np.testing.assert_allclose(metrics["theta_row_sums_mean"], 1.0, atol=1e-4)
    np.testing.assert_allclose(metrics["beta_row_sums_mean"], 1.0, atol=1e-4)


def test_toy_pipeline_heldout_negative():
    metrics = _run_toy_pipeline()
    # With random init and only 2 EM iterations, some beta columns may be zero,
    # causing log(0) = -inf. We accept -inf as a valid negative value here.
    assert metrics["heldout_ll"] < 0 or metrics["heldout_ll"] == float("-inf")


@pytest.mark.skipif(
    not os.path.exists(FIXTURE_PATH),
    reason="Baseline fixture not recorded yet. Run: uv run python tests/record_baseline.py",
)
def test_baseline_regression():
    """Compare current output against recorded baseline metrics."""
    with open(FIXTURE_PATH) as f:
        baseline = json.load(f)

    metrics = _run_toy_pipeline()

    # Shape checks (exact)
    assert metrics["beta_shape"] == baseline["beta_shape"]
    assert metrics["theta_shape"] == baseline["theta_shape"]
    assert metrics["sigma_shape"] == baseline["sigma_shape"]

    # Numeric checks (tolerance)
    np.testing.assert_allclose(
        metrics["final_bound"], baseline["final_bound"], rtol=0.01,
        err_msg="final_bound drifted >1% from baseline",
    )

    # heldout_ll may be -inf if beta has zero columns (random init, few iterations)
    if np.isfinite(baseline["heldout_ll"]):
        np.testing.assert_allclose(
            metrics["heldout_ll"], baseline["heldout_ll"], rtol=0.05,
            err_msg="heldout_ll drifted >5% from baseline",
        )
    else:
        assert metrics["heldout_ll"] == baseline["heldout_ll"]
