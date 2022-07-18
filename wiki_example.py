#%%
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
from heldout import find_k 
import numpy as np
import pandas as pd 
from stm import STM


# load corpus & dictionary
data = pd.read_csv('application/data/corpus_preproc.csv')
corpus = corpora.MmCorpus('application/data/BoW_corpus.mm')
dictionary = corpora.Dictionary.load('application/data/dictionary.mm')
#%%
#set configuration
K=10
kappa_interactions = False
lda_beta = True
beta_index = None
max_em_iter = 10
sigma_prior = 0
convergence_threshold = 0.25
# set topical prevalence 
prevalence = ['statistics','ml']
xmat = data.loc[:, prevalence]
stm_config = {
    "content": False,
    "K": K,
    "kappa_interactions": False,
    "lda_beta": True,
    "max_em_iter": max_em_iter,
    "sigma_prior": sigma_prior,
    "convergence_threshold": convergence_threshold,
    "init_type": "spectral",
    "model_type":"STM",
}

model = STM(documents=corpus, dictionary=dictionary, X=xmat, **stm_config)
model.expectation_maximization(saving=False)

# %% investigate topics (highest probable words)
model.label_topics(n=30, topics=10)
#%% investigate covariate effect on topics
model.gamma
# %%
