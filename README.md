# strutopy 

Is a Python-package focussing on the Structural Topic Model and machine-assisted reading of large text corpora. 
The implementation in Python aims for computational efficiency as well as ease-of-use.

Structural Topic Model (Roberts et al. 2014) can be used to extend the classical topic modelling approaches by including text metadata on a document level. 
The meta information can be introduced to the estimation procedure two-fold, via: 
1. topical content covariates that shape the word usage within topics
2. topical prevalence covariates that shape the frequency of topic occurences.

## Package Structure 

The packages consists of three main parts: 

1. Text Reading for various filetypes (*.csv, *.json) 

2. Text Preparation
- Pre-processing
  - Stopword-Removal
  - Stemming
  - Dropping Documents
  - Removing Punctuation 
  - **n-gram** algorithm
- Corpus creation
  - list of documents containing word indices and their count
  - vector of words associated with the indices
  - metadata matrix with document covariates

3. Model Estimation
- Spectral Initialisation
- Topical Prevalence Model
  - interaction terms, standard transforms and non-linear relations, such as splines
- Topical Content Model 

4. Model Evaluation
- Semantic Coherence Measure: Goodness-of-topics depends on whether most probable words in a given topic frequently co-occur together
- Exclusivity: Word-exclusivity on a topic level 
- FREX: harmonic mean for semantic coherence and exclusivity

5. Visualisation
- Corpus visualisations: 
  - wordclouds
  - word frequencies
  - tf-idf -> t-sne visualisation
- Estimate visualisation: 
  - Metadata estimates can be visualised w.r.t. their effect on the expected topic proportions as well as on the topical content
  - Visualisation of identified topics and their distances in a topic - graph 
