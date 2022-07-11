#%%
import pandas as pd
import numpy as np
from gensim import corpora
import time
import math
from stm import STM

#%% load data
# raw documents
data = pd.read_csv('data/poliblogs2008.csv')
# selection for quick testing

# load preprocessed corpus
documents = corpora.MmCorpus('data/corpus.mm')
# dictionary: 
dictionary = corpora.Dictionary.load('data/dictionary')
# vocabulary: 
vocab = dictionary.token2id

#%% """ Setting control variables"""
prevalence = 'rating'
content = 'blog'
num_topics = 10

#Initialize parameters
data = data[3000:4000]
xmat = data.loc[:, prevalence]
yvar = data.loc[:, content].astype('category')
yvarlevels = set(yvar)
betaindex = np.array(yvar.cat.codes)
A = len(set(betaindex))

interactions = False #settings.kappa
verbose = True

init_type = "Random" #settings.init
ngroups = 1 #settings.ngroups
max_em_its = 5 #settings.convergence
emtol = 1e-5 #settings.convergence
sigma_prior=0 #settings.sigma.prior


settings = {
    'dim':{
        'K': num_topics, #number of topics
        'V' : len(dictionary), #number of words
        'A' : A, #dimension of topical content
        'N' : len(documents),
    },
    'verbose':verbose,
    'kappa':{
        'interactions':False,
        'fixedintercept': True,
        'contrats': False,
        'mstep': {'tol':0.01, 'maxit':5}},
    'tau':{
        'mode': np.nan,
        'tol': 1e-5,
        'enet':1,
        'nlambda':250,
        'lambda.min.ratio':.001,
        'ic.k':2,
        'maxit':1e4},
    'init':{
        'mode':init_type, 
        'nits':20,
        'burnin':25,
        'alpha':50/num_topics,
        'eta':.01,
        's':.05,
        'p':3000},
    'convergence':{
        'max.em.its':max_em_its,
        'em.converge.thresh':emtol,
        'allow.neg.change':True,},
    'covariates':{
        'X':xmat,
        'betaindex':betaindex,
        'yvarlevels':yvarlevels,
        'formula': prevalence,},
    'gamma':{
        'mode':'L1', #needs to be set for the m-step (update mu in the topical prevalence model)
        'prior':np.nan, #sigma in the topical prevalence model
        'enet':1, #regularization term
        'ic.k':2,#information criterion
        'maxits':1000,},
    'sigma':{
        'prior':sigma_prior,
        'ngroups':ngroups,},
}

#%%
model = STM(settings, documents, dictionary)
model.expectation_maximization(saving=True)
