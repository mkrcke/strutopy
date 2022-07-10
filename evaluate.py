#%%
import json
import matplotlib.pyplot as plt
import numpy as np 

from generate_docs import CorpusCreation
# load model
with open('stm_model_2.json') as f:
    stm_model = json.load(f)

#%% extract metrics from model result    
bound = stm_model['bound']
iterations = range(len(bound))
plt.plot(iterations, bound)
plt.grid(which='both')
plt.show()
#%%
stm_model['lambda']
# %%
Corpus = CorpusCreation(
    n_topics=stm_model['settings']['dim']['K'],
    n_docs=100,
    n_words=100,
    V=stm_model['settings']['dim']['V'],
    treatment=False,
    theta=stm_model['lambda'],
    beta=stm_model['beta'],
)
Corpus.generate_documents()
# %%