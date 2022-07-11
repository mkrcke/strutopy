#%%
import json
import matplotlib.pyplot as plt
import numpy as np 

from generate_docs import CorpusCreation
# load model
with open('model.json') as f:
    stm_model = json.load(f)

#%% extract metrics from model result    
bound = [-3873893.8525840244, -2922256.22750034, -2923069.9152508364, -2923360.738664682, -2923615.1195009947, -2923802.2893727114, -2923752.269674102, -2923388.3467372633, -2922966.893290721, -2922396.187998074, -2921745.8607996637, -2921019.086945867, -2920187.0568499276, -2919265.344560678, -2918199.8425150653, -2916974.9100953517, -2915599.863686228, -2914114.0253530005, -2912543.5091583543, -2910847.031035186, -2909043.5566455713, -2907129.8139873384, -2905140.332832266, -2903092.380413262, -2900967.7121907086, -2898812.9792277073, -2896604.156755537, -2894361.3309212765, -2892080.549206204, -2889776.2419606233]
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