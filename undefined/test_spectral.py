#%%
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from qpsolvers import solve_qp
from scipy.sparse import diags, csr_matrix, csr_array
from sklearn.preprocessing import normalize
from spectral_initialisation import spectral_init
from gensim import corpora
#%% import data
data = pd.read_csv('data/poliblogs2008.csv')
corpus = corpora.MmCorpus('data/corpus.mm')
dictionary = corpora.Dictionary.load('data/dictionary')
#%%
K=10
beta = spectral_init(corpus, V = len(dictionary), maxV=2000, verbose=True, K=K)

# %%
