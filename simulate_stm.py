#%% import packages
import numpy as np
from generate_docs import CorpusCreation
from spectral_initialisation import spectral_init
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from stm import STM


#%% Simulate a corpus
np.random.seed(12345)
Corpus = CorpusCreation(
    n_topics = 10,
    n_docs=400,
    n_words=100,
    V=2500,
    treatment=False,
    alpha='symmetric')

Corpus.generate_documents()
#sample unique tokens to build the dictionary
print('Number of unique tokens: %d' % len(Corpus.dictionary))
print('Number of documents: %d' % len(Corpus.documents))

#%%
# Set starting values and parameters
# Parameter Settings (required for simulation process)
num_topics = 10
A = 2
verbose = True
interactions = False  # settings.kappa
betaindex = np.concatenate(
    [np.repeat(0, len(Corpus.documents) / 2), np.repeat(1, len(Corpus.documents) / 2)]
)


# Initialization and Convergence Settings
ngroups = 1  # settings.ngroups
max_em_its = 15  # settings.convergence
emtol = 1e-5  # settings.convergence


settings = {
    "dim": {
        "K": num_topics,  # number of topics
        "V": len(Corpus.dictionary),  # number of words
        "A": A,  # dimension of topical content
        "N": len(Corpus.documents),
    },
    "verbose": verbose,
    "kappa": {
        "interactions": interactions,
        "fixedintercept": True,
        "contrats": False,
        "mstep": {"tol": 0.01, "maxit": 5},
    },
    "tau": {
        "mode": np.nan,
        "tol": 1e-5,
        "enet": 1,
        "nlambda": 250,
        "lambda.min.ratio": 0.001,
        "ic.k": 2,
        "maxit": 1e4,
    },
    "init": {
        "mode": init_type,
        "nits": 20,
        "burnin": 25,
        "alpha": 50 / num_topics,
        "eta": 0.01,
        "s": 0.05,
        "p": 3000,
    },
    "convergence": {
        "max.em.its": max_em_its,
        "em.converge.thresh": emtol,
        "allow.neg.change": True,
    },
    "covariates": {
        "X": betaindex,
        "betaindex": betaindex,
        #     'yvarlevels':yvarlevels,
        #     'formula': prevalence,
    },
    "gamma": {
        "mode": "L1",  # needs to be set for the m-step (update mu in the topical prevalence model)
        "prior": np.nan,  # sigma in the topical prevalence model
        "enet": 1,  # regularization term
        "ic.k": 2,  # information criterion
        "maxits": 1000,
    },
    "sigma": {
        "prior": 0,
        "ngroups": ngroups,
    },
}

#%% compare random vs. spectral
bounds = np.empty(2, dtype=list, order='C')
for i, init in enumerate(['random', 'spectral']):
    model = STM(settings, Corpus.documents, Corpus.dictionary, init=init) 
    model.expectation_maximization(saving=False)
    bounds[i] = model.last_bounds



# %% plot bounds
import matplotlib.pyplot as plt
x = np.arange(max_em_its)
y1 = bounds[0]
y2 = bounds[1]
# plotting
plt.plot(x, y1, label = "random initialization")
plt.plot(x, y2, label = "spectral initialization")
plt.xlabel('#iteration')
plt.ylabel('ELBO')
plt.title('Random vs. Spectral Initialization')
plt.legend()
plt.savefig('img/spectral_init.png', bbox_inches='tight', dpi=360)
plt.show()
# %%
