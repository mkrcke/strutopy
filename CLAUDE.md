# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Strutopy is a Python implementation of Structural Topic Modeling (STM) for machine-assisted reading of large text corpora. It extends classical topic modeling by incorporating document-level metadata through topical content covariates (shaping word usage within topics) and topical prevalence covariates (shaping topic frequency). Based on Roberts et al. (2014).

## Setup

```bash
uv sync          # create .venv and install all dependencies
uv add <pkg>     # add a new dependency
```

Package is managed via `uv` with `pyproject.toml`. Python >=3.12.

## Pipeline

The project follows a sequential numbered pipeline in `src/`:

1. `01_get_wiki_docs.py` — Fetch Wikipedia documents
2. `02_create_corpus.py` — Preprocess text and create corpus
3. `03_fit_reference_model.py` — Train reference STM model
4. `04_create_synthetic_corpora.py` — Generate synthetic corpora
5. `05_train.py` — Train models on synthetic data
6. `06_example_application.py` — Example usage

Run pipeline scripts individually (`python src/05_train.py`) or via `script.sh`.

## Architecture

### Core Modules (`src/modules/`)

- **`stm.py`** — Main STM implementation (~1250 lines). Contains:
  - `STM` class: model fitting via Expectation-Maximization
  - `spectral_init()`: deterministic spectral initialization (Arora et al. 2014), recommended for >40k docs
  - `create_dtm()`, `gram()`, `fastAnchor()`, `recover_l2()`: supporting functions for initialization
- **`generate_docs.py`** — `CorpusCreation` class for synthetic data generation following LDA/STM data generating process
- **`heldout.py`** — Evaluation metrics: semantic coherence, exclusivity, FREX, held-out likelihood
- **`chunk_it.py`** — Utility for data chunking

### Data Flow

- Gensim corpus format (.mm files) for document-term matrices
- Numpy arrays (.npy) for model parameters
- Model artifacts stored in `src/artifacts/` (gitignored)
- Logging to `logfiles/` directory

### Key Patterns

- `joblib.Parallel` for multi-core model training
- Gensim `Dictionary` and `MmCorpus` for corpus representation
- `qpsolvers` for quadratic programming in L2 recovery
- Notebooks in `notebooks/` for experimentation and visualization