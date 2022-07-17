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

# set topical prevalence 
prevalence = ['statistics','ml']
xmat = data.loc[:, prevalence]

settings = {
    'dim':{
        'K': 20,
        'V' : len(dictionary), 
        'N' : len(corpus),
        'A' : None
    },
    'verbose':True,
    'kappa':{
        'LDAbeta': True,
        'interactions':False,
        'fixedintercept': True,
        'contrast': False,
        'mstep': {'tol':0.01, 'maxit':5}},
    'convergence':{
        'max.em.its':6,
        'em.converge.thresh':1e-5}
    'covariates':{
        'X':xmat,
        'betaindex': None}, # this is were the topical prevalence covariates are included 
    'gamma':{
        'prior':np.nan, #sigma in the topical prevalence model
        'enet':1, #regularization term
        'ic.k':2,#information criterion
        'maxits':1000,},
    'sigma':{
        'prior':0,
        'ngroups':1,},
}
# %%
model = STM(settings, corpus, dictionary, init='random', model='STM', content = False)
model.expectation_maximization(saving=False, prefix='wiki')
# %% FREX score
model.label_topics(n=20, topics=20)
