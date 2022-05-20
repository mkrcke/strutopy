#%%
import numpy as np
import math
import pandas as pd
from gensim import corpora
from scipy import optimize
from init import random_init, init_kappa
from settings import settings
#%% 
data = pd.read_csv('poliblogs2008.csv')

#%%
documents = corpora.MmCorpus('corpus.mm')
dictionary = corpora.Dictionary.load('dictionary')
vocab = dictionary.token2id

#%% load settings
K = 10 #settings.dim
V = len(dictionary) #settings.dim
N = len(documents) #settings.dim

interactions = True #settings.kappa
verbose = True

init_type = "Random" #settings.init
ngroups = 1 #settings.ngroups
max_em_its = 15 #settings.convergence
emtol = 1e-5 #settings.convergence
settings[]