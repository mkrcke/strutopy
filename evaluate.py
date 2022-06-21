import json
import matplotlib.pyplot as plt
import numpy as np 
# load model
with open('stm_model.json') as f:
    stm_model = json.load(f)

# extract metrics from model result    
bound = stm_model['convergence']['bound']
iterations = range(len(bound))
plt.plot(iterations, bound)
plt.show()

stm_model['mu']
np.log(stm_model['beta'])
stm_model['sigma']

