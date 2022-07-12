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