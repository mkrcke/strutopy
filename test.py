# %%

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import json

# custom packages

from stm import STM
from simulate import generate_docs


def basic_simulations(n_docs, n_words, V, ATE, alpha, display=True):
    generator = generate_docs(n_docs, n_words, V, ATE, alpha)
    documents = generator.generate(n_docs)
    if display == True:
        generator.display_props()
    return documents

# Here we are simulating 100 documents with 100 words each. We are sampling from a multinomial distribution with dimension V.
# Note however that we will discard all elements from the vector V that do not occur.
# This leads to a dimension of the vocabulary << V
np.random.seed(123)
documents, vocabulary = basic_simulations(n_docs=100, n_words=40, V=500, ATE=.2, alpha=np.array([.3,.4,.3]), display=False)
betaindex = np.concatenate([np.repeat(0,50), np.repeat(1,50)])
num_topics = 3
dictionary=np.arange(vocabulary)
# %%
