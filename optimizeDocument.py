#%%
import numpy as np
import math
import pandas as pd
import gensim 
from gensim import corpora
import scipy
from scipy import optimize
from init import *
#%% 
data = pd.read_csv('poliblogs2008.csv')

#%%
documents = corpora.MmCorpus('corpus.mm')
dictionary = corpora.Dictionary.load('dictionary')
vocab = dictionary.token2id