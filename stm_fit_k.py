# %%
from gensim.corpora.dictionary import Dictionary
from heldout import find_k 
import numpy as np

#%% 
# load corpus & dictionary



def stm_fit_k():
    K_candidates = np.array([5,10,20,30])
    results_k = find_k(K_candidates, corpus, settings)
    # define k candidates
    