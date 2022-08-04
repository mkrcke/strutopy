# strutopy: Python Implementation for the Structural Topic Model

The implementation in Python aims for computational efficiency as well as ease-of-use.

Structural Topic Model (Roberts et al. 2016) can be used to extend the former topic modelling approaches by including text metadata on a document level. 
The meta information can be introduced to the estimation procedure two-fold, via: 
1. topical content covariates that shape the word usage within topics
2. topical prevalence covariates that shape the frequency of topic occurences.

## Contribution of this thesis: 
The contribution of this thesis is twofold: 
1. A complete implementation of the STM model and estimation procedure is provided. The main module is contained in src/modules/stm.py
2. A comprehensive comparison between STM and its direct predecessor CTM is conducted. 

## Setup
In order to reproduce and use the code of the repository, the following steps are recommended:

Create and activate a new virtual environment (from the root of the repository):
```
python -m venv .venv
```

Activate the virtual environment:
```
source .venv/bin/activate
```

Install all python dependencies:
```
pip install -r requirements.txt
```

## Quickstart

Navigate to the following folder:
```
cd src
```

And execute the python scripts in the following order:

1. To get the articles from wikipedia, run src/01_get_wiki_docs.py:
```
python 01_get_wiki_docs.py
```

2. To create the reference corpus, run src/02_create_corpus.py:
```
python 02_create_corpus.py
```

3. To fit the reference model for K=[10,...,70], run src/03_fit_reference_model.py:
```
python 03_fit_reference_model.py
```

4. To create the set of synthetic corpora for varying value of gamma, run src/04_create_synthetic_corpora.py:
```
python 04_create_synthetic_corpora.py
```

5. To train STM & CTM on the synthetic corpora, run src/05_train.py:
```
python 05_train.py
```
Training STM & CTM provides us with the simulation results obtained in this master thesis. The following example visualisation can be obtained: 
![example results](img/3_5_bp.png?raw=true "Example results for the comparison of STM and CTM.")

6. To reproduce the application results, run src/06_example_application.py:
```
python 06_example_application.py
```
## Repository Components

1. Corpus creation: A corpus in the Bag-of-Words format is required for STM. In addition, metadata is needed to play to the strengths of STM:
  - list of documents containing word indices and their count
  - vector of words associated with the indices
  - metadata matrix with document covariates

2. Model Estimation (src/modules/stm.py)
- Spectral Initialisation
- Semi collapsed Laplace Approximation Variational Expectation-Maximization Algorithm
- Topical Prevalence Model
- Topical Content Model 

3. Model Evaluation
- Label topics by retrieving the most relevant words per topic. Relevance is given in terms of:
  - Highest probability
  - FREX: harmonic mean for semantic coherence and exclusivity

