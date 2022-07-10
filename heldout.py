# The heldout likelihood is a measure of predictive performance
# Based on fitted parameter values, we want to maximise the probability of unseen word (or documents)
# The Likelihood is defined as P(W"|W',theta, beta)

# Computing the predictive likelihood for unseen documents requires reestimation (not necessarily)
# Computing the predictive likelihood for unseen words in a document used for training does not require reestimation 
# -> Document Completion Method

# Document Completion Approach
# 0. Estimate beta on the whole corpus
# Generate a test, training and validation set based on a corpus (80/10/10)
# 1. Remove a certain portion of each document (~50%)
# 2. Fit STM on the remaining words
# 3. Compute the likelihood for the test portion based on the trained model parameters (theta, beta) 
# 4. Average over words in a document to get the held-out likelihood for a single document
# 5. Average over each documents held-out likelihood

# Requirements. 
## STM.expectation_maximization() returns the document-level mean posterior values (theta, beta) - CHECK 
## A function that can split the corpus into test, train and validate
## We can introduce fitted parameters (theta, beta) into the document for computing the heldout-likelihood. 


#%% Load packages and model
import json
import matplotlib.pyplot as plt
import numpy as np 
from stm import STM
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from generate_docs import CorpusCreation

#%%
def split_corpus(corpus, proportion=0.8): 
  test_split_idx = int(proportion*len(corpus))
  validate_split_idx = int((proportion+(1-proportion)/2)*len(corpus))

  train = np.array(corpus[:test_split_idx])
  test = np.array(corpus[test_split_idx:validate_split_idx])
  validate = np.array(corpus[validate_split_idx:])

  return train, test, validate

def cut_in_half(set):
  """function to split a set of documents in two parts

  Args:
      set (np.ndarray): set of documents in bag-of-words format

  Returns:
      first_half: returns the set with every other word removed (starting at index 0)
      second_half: returns the set with every other word removed (starting at index 1)
  """
  first_half = np.empty(len(set), dtype=np.ndarray)
  second_half = np.zeros(len(set), dtype=np.ndarray)
  
  for doc in range(len(set)): 
    first_half[doc] = (set[doc][0::2])
    second_half[doc] = (set[doc][1::2])

  return first_half, second_half

def eval_heldout(heldout, theta, beta):
  doc_ll=[]
  for i,doc in enumerate(heldout):
    word_ll = []
    for word in doc:  
      word_ll.append(word[1]*np.log(theta[i]@beta[:,word[0]]))
    per_word_ll = np.sum(word_ll)/np.sum(np.array(doc)[:,1])
    doc_ll.append(per_word_ll)
  
  return doc_ll, np.mean(doc_ll)


def train_models(beta_train_corpus, theta_train_corpus, K):
  settings["dim"]["K"] = K
  # initialize dictionaries for different corpora
  model_beta_dictionary = Dictionary.from_corpus(beta_train_corpus)
  model_theta_dictionary = Dictionary.from_corpus(theta_train_corpus)
  # initialize models
  model_beta = STM(settings, beta_train_corpus, model_beta_dictionary) 
  model_theta = STM(settings, theta_train_corpus, model_theta_dictionary)
  # take beta from model trained on train + test set
  model_beta.expectation_maximization(saving=False)
  # take theta from model trained on train + half of test documents
  model_theta.expectation_maximization(saving=False)

  return np.array(model_beta.beta), np.array(model_theta.lamda)

def heldout(corpus, K):
  
  train, test, validate = split_corpus(corpus, proportion=0.8)
  test_1, test_2 = cut_in_half(test)
  # validate_1, validate_2 = cut_in_half(validate)

  beta_train_corpus = np.concatenate((train, test))
  theta_train_corpus = np.concatenate((train, test_1))

  beta, theta = train_models(beta_train_corpus, theta_train_corpus, K=K)
  
  doc_ll, expected_ll = eval_heldout(heldout=test_2, beta=beta, theta=theta)

  return doc_ll, expected_ll


def find_k(candidates, corpus, settings):
  results=[]
  for K in candidates:
    _,expected_ll = heldout(corpus, K=K)
    results.append(expected_ll)
  
  return results

#%%
Corpus = CorpusCreation(
    n_topics=50,
    n_docs=500,
    n_words=50,
    V=1500,
    treatment=False,
    alpha='symmetric',
)
Corpus.generate_documents()
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
init_type = "Random"  # settings.init
ngroups = 1  # settings.ngroups
max_em_its = 5  # settings.convergence
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


# %%

corpus = Corpus.documents
candidates = np.array([3,10,30,50,100])
results_k = find_k(candidates, corpus, settings)

# %% plot
fig, ax = plt.subplots()

ax.scatter(candidates, results_k)


plt.title("Held-out Likelihood for varying number of topics")
plt.xlabel("Number of topics")
plt.ylabel("Held-out Likelihood")
plt.show()
# %%
