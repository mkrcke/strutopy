#%%
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
from heldout import find_k 
import numpy as np
import pandas as pd 
from stm import STM


# load corpus & dictionary
data = pd.read_csv('artifacts/wiki_data/corpus_preproc.csv')
corpus = corpora.MmCorpus('artifacts/wiki_data/BoW_corpus.mm')
dictionary = corpora.Dictionary.load('artifacts/wiki_data/dictionary.mm')
#%%
#set configuration
K=20
kappa_interactions = False
lda_beta = True
beta_index = None
max_em_iter = 30
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
K = 10
prob, frex = model.label_topics(n=20, topics=range(K))
# investigate covariate effect on topics
for topic in range(K-1): 
    print(f"Statistics: {round(model.gamma[topic][0],4)} * {frex[topic]})")
    print(f"ML:  {round(model.gamma[topic][1],4)} * {frex[topic]} \n")

# %%
