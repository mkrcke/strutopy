#%%
import json
import matplotlib.pyplot as plt
import numpy as np 
# load model
with open('stm_model_2.json') as f:
    stm_model = json.load(f)

#%% extract metrics from model result    
bound = stm_model['bound']
iterations = range(len(bound))
plt.plot(iterations, bound)
plt.show()
#%%
stm_model['mu']
np.log(stm_model['beta'])
stm_model['sigma']
# %% compute mean topic proportion (theta) 
np.exp(stm_model['mu'])/np.sum(np.exp(stm_model['mu']))
# %%
