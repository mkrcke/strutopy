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
    },
    'verbose':True,
    'kappa':{
        'interactions':False,
        'fixedintercept': True,
        'contrats': False,
        'mstep': {'tol':0.01, 'maxit':5}},
    'convergence':{
        'max.em.its':5,
        'em.converge.thresh':1e-5,
        'allow.neg.change':True,},
    'covariates':{
        'X':xmat,}, # this is were the topical prevalence covariates are included 
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
model = STM(settings, corpus, dictionary, init='spectral', model='STM')
model.expectation_maximization(saving=True, prefix='wiki')
# %%
