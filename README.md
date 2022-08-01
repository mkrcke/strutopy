# strutopy: Python Implementation for the Structural Topic Model

The implementation in Python aims for computational efficiency as well as ease-of-use.

Structural Topic Model (Roberts et al. 2016) can be used to extend the classical topic modelling approaches by including text metadata on a document level. 
The meta information can be introduced to the estimation procedure two-fold, via: 
1. topical content covariates that shape the word usage within topics
2. topical prevalence covariates that shape the frequency of topic occurences.

## Usage example



## Main components of the repository

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
