# Modules — Implementation Reference

This document describes the core modules implementing Structural Topic Modeling (STM) based on Roberts et al. (2014), with spectral initialization from Arora et al. (2014).

**Important**: The mathematical formulas in this codebase have been carefully developed and validated. Do NOT modify numerical routines (normalization, optimization objectives, Hessian computation, etc.) unless a bug is conclusively proven with a failing test case or reproducible incorrect output.

## File Overview

| File | Purpose |
|---|---|
| `stm.py` | Main STM: spectral init, EM algorithm, variational inference |
| `generate_docs.py` | Synthetic corpus generation (LDA/STM data generating process) |
| `heldout.py` | Evaluation: held-out likelihood, semantic coherence, FREX |
| `chunk_it.py` | Utility for partitioning sequences into chunks |

---

## stm.py — Core STM Implementation (~1250 lines)

### Spectral Initialization Pipeline

Called when `init_type="spectral"`. Deterministic (no randomness). Recommended for D > 40,000.

**Flow**: `spectral_init()` → `create_dtm()` → `gram()` → `fastAnchor()` → `recover_l2()`

#### `spectral_init(corpus, K, V, maxV=5000, verbose=True, print_anchor=False)`
1. Builds document-term matrix from BoW corpus
2. Computes empirical word probabilities `wprob`
3. Filters to top `maxV` words (controls O(V^2) complexity)
4. Computes gram matrix Q (word co-occurrence)
5. Finds K anchor words via greedy projection
6. Recovers beta via constrained L2 optimization
7. Re-expands beta to full vocabulary with pseudocount `0.001/V`

#### `create_dtm(corpus)`
Converts BoW corpus `[(word_idx, count), ...]` to scipy `csr_matrix` (D x V).

#### `gram(doc_term_matrix)`
Computes row-normalized co-occurrence matrix Q (V x V). Uses sparse arithmetic. Asserts all row sums > 0 before normalizing via `sklearn.preprocessing.normalize`.

#### `fastAnchor(Q, K, verbose=True)`
Greedy anchor word selection. Iteratively finds the word with largest projected norm, normalizes, projects remaining words, and repeats K times. Transitions Q from sparse to dense during iteration.

#### `recover_l2(Q, anchor, wprob)`
For each non-anchor word, solves a QP to find the convex combination of anchor rows that best reconstructs that word's row in Q. Uses `qpsolvers` with the `quadprog` solver. Transforms P(z|w) to P(w|z) via Bayes' rule using `wprob`.

### STM Class

#### Constructor Key Parameters
- `documents`: BoW corpus `[[(word_idx, count), ...], ...]`
- `dictionary`: Gensim `Dictionary`
- `content` (bool): enable topical content model (beta varies by aspect)
- `K`: number of topics
- `X`: topical prevalence covariates (N x p)
- `beta_index`: topical content covariate indices
- `A`: number of aspect levels
- `init_type`: `"spectral"` or `"random"`
- `model_type`: `"STM"` or `"CTM"`
- `mode`: `"ols"`, `"ridge"`, or `"lasso"` for prevalence estimation
- `sigma_prior`: weight for diagonal covariance prior (0=MLE, 1=diagonal only)
- `lda_beta` (bool): row-normalize beta (True) vs. Poisson regression (False)

#### Key Dimensions
- `beta`: K x V (or A x K x V with content model)
- `theta`: N x K (document-topic proportions, simplex)
- `eta`: N x (K-1) (unconstrained logistic-normal parameters; K-th component implicitly 0)
- `mu`: N x (K-1) (mean of logistic normal, predicted from covariates)
- `sigma`: (K-1) x (K-1) (covariance of logistic normal)

#### EM Algorithm (`expectation_maximization`)
```
for each iteration:
    E-step: optimize variational parameters per document, compute ELBO
    M-step: update mu (prevalence), sigma (covariance), beta (word-topic)
    Check convergence: |ELBO_new - ELBO_old| / |ELBO_old| < threshold
```

#### E-Step Detail
For each document:
1. Extract document's words and select relevant beta columns
2. Optimize eta via BFGS (`optimize_eta`) — minimizes variational objective
3. Softmax eta → theta (with K-th component = 0)
4. Compute negative Hessian of variational objective
5. Cholesky decompose Hessian (with PD fallbacks)
6. Compute ELBO contribution
7. Update sufficient statistics: `sigma_ss`, `beta_ss`

#### Variational Objective (`optimize_eta`)
```
f(eta) = 0.5 * (eta - mu)^T @ sigma_inv @ (eta - mu)
         - sum_v(count_v * logsumexp(eta + log(beta[:, v])))
         + N_doc * logsumexp(eta)
```
Optimized with `scipy.optimize.minimize(method='BFGS')`.

#### M-Step Detail
- **`update_mu`**: Regress eta on covariates X (OLS/Ridge/Lasso). CTM mode uses global mean.
- **`update_sigma`**: Empirical covariance + diagonal shrinkage controlled by `sigma_prior`.
- **`update_beta`**: Row-normalize `beta_ss` (LDA-style) or fit Poisson regression (`mnreg`) for content model.

#### Numerical Stability
- Stable softmax: subtracts max before exp
- Cholesky fallbacks: diagonal dominance → add small constant → scipy with regularization
- `logsumexp` from scipy.special throughout
- PD check before Cholesky; `make_pd()` adjusts diagonal if needed

#### Topic Labeling (`label_topics`)
- **Highest Probability**: top-n words by P(w|z)
- **FREX**: harmonic mean of frequency rank and exclusivity rank, weighted by `frexweight`

---

## generate_docs.py — Synthetic Corpus Generation (~418 lines)

### `CorpusCreation` Class

Simulates documents under LDA or STM generative processes for model evaluation.

#### Key Parameters
- `n_topics`, `n_docs`, `n_words`, `V`: corpus dimensions
- `dgp`: `"STM"` or `"LDA"` data generating process
- `alpha`: Dirichlet prior — `"symmetric"` (1/K each) or `"asymmetric"` (1/(k + sqrt(k)))
- `treatment` (bool): split documents into control/treatment groups with different priors
- `level`: number of covariate levels
- `metadata`: document-level covariates (N x level)

#### Generation Flow
1. `init_alpha()` — set Dirichlet prior(s)
2. `word_topic_dist()` — sample beta from Dir(0.05) or use provided
3. `init_gamma()` — sample prevalence coefficients from N(0, I)
4. `set_metadata()` — assign categorical covariates
5. `init_eta()` — sample eta from N(metadata @ gamma, 0.001*I)
6. `init_theta()` — LDA: sample from Dirichlet; STM: softmax(eta)
7. `generate_documents()` → `sample_documents()` — draw words from Multinomial(n_words, theta @ beta)

#### Corpus Splitting (`split_corpus`)
- Train 80% / Test 10% / Validate 10%
- Document completion: split test docs into even/odd word indices

---

## heldout.py — Evaluation Metrics (~156 lines)

### `eval_heldout(heldout, theta, beta)`
Per-document average log-likelihood:
```
L(d) = sum_v(count_v * log(theta[d] @ beta[:, v])) / sum_v(count_v)
```
Returns mean over all documents.

### `heldout_on_test(corpus, K, model, settings)`
Full pipeline: split → train beta on (train+test) → train theta on (train+test_1) → evaluate on test_2.

### `find_k(K_candidates, models, corpus, settings)`
Grid search over topic numbers and model types. Returns held-out likelihood matrix.

---

## chunk_it.py — Utility (~14 lines)

### `chunkIt(seq, num)`
Partitions a sequence into `num` roughly equal chunks using rounding-based index slicing. Used for parallel processing splits.

---

## Dependencies

| Package | Usage |
|---|---|
| `scipy.sparse` | CSR matrices for document-term and gram matrices |
| `scipy.optimize` | BFGS for variational eta optimization |
| `scipy.special` | `logsumexp` for numerical stability |
| `qpsolvers` | Quadratic programming in `recover_l2` (solver: `quadprog`) |
| `sklearn` | Linear models (Ridge, Lasso, Poisson), OneHotEncoder, normalize |
| `gensim` | Dictionary, MmCorpus for corpus I/O |
| `numpy` | Core numerical operations |

## References

- Roberts, Stewart & Tingley (2016) — STM methodology
- Arora et al. (2014) — Spectral learning / anchor words
- Bischof & Airoldi (2012) — FREX metric
- Blei, Ng & Jordan (2003) — LDA
