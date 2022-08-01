#%%
import pandas as pd
import numpy as np
from gensim import corpora
import time
import math
from stm import STM

#%% load data
# load preprocessed corpus
documents = corpora.MmCorpus('data/corpus.mm')
# dictionary: 
dictionary = corpora.Dictionary.load('data/dictionary')
# vocabulary: 
vocab = dictionary.token2id

# load metadata matching to the documents 3000-4000
data = pd.read_csv('data/poliblogs2008.csv')
data = data[3000:4000]

#%% """ Setting control variables"""
prevalence = ['rating','blog']
content = 'blog'
num_topics = 10

#Initialize parameters

xmat = data.loc[:, prevalence]
yvar = data.loc[:, content].astype('category')
yvarlevels = set(yvar)
betaindex = np.array(yvar.cat.codes)
A = len(set(betaindex))

interactions = True #settings.kappa
verbose = True

init_type = "random" #settings.init
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
        'LDAbeta': False, 
        'interactions':True,
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
        'prior':np.nan, #sigma in the topical prevalence model
        'enet':1, #regularization term
        'ic.k':2,#information criterion
        'maxits':1000,},
    'sigma':{
        'prior':sigma_prior,
        'ngroups':ngroups,},
}

#%%
model = STM(settings, documents, dictionary, init='random', model='STM')
model.expectation_maximization(saving=False, prefix='')

# %%
