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
## A function that can split the corpus into test, train and validate - CHECK
## We can introduce fitted parameters (theta, beta) into the document for computing the heldout-likelihood - CHECK


import inspect

#%% Load packages and model
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim import utils
from gensim.corpora.dictionary import Dictionary

from .generate_docs import CorpusCreation
from .stm import STM


#%%
def split_corpus(corpus, validation_set=False, document_completion=True, proportion=0.8):
        
        test_split_idx = int(proportion * len(corpus))
        
        if not type(corpus)==list:
            try: 
                train = [doc for doc in corpus[:test_split_idx]]
                test = [doc for doc in corpus[test_split_idx:validate_split_idx]]
                validate = [doc for doc in corpus[validate_split_idx:]]
            except:
                corpus = [doc for doc in corpus] 
                train_docs = corpus[:test_split_idx]
                test_docs = corpus[test_split_idx:]

        if validation_set:
            validate_split_idx = int(
                (proportion + (1 - proportion) / 2) * len(corpus)
            )
            test_docs = corpus[test_split_idx:validate_split_idx]
            validate_docs = corpus[validate_split_idx:]

        else:
            test_docs = corpus[test_split_idx:]

        if document_completion:
            test_1_docs, test_2_docs = cut_in_half(test_docs)
        
        return train_docs, test_1_docs, test_2_docs, validate_docs


def cut_in_half(doc_set):
    """function to split a set of documents in two parts

    @param: doc_set (np.ndarray) set of documents in bag-of-words format

    @return: first_half returns the set with every other word removed (starting at index 0)
    @return: second_half returns the set with every other word removed (starting at index 1)
    """
    first_half = np.zeros(len(doc_set), dtype=np.ndarray)
    second_half = np.zeros(len(doc_set), dtype=np.ndarray)

    for doc in range(len(doc_set)):
        first_half[doc] = doc_set[doc][0::2]
        second_half[doc] = doc_set[doc][1::2]

    return first_half, second_half


def eval_heldout(heldout, theta, beta):
    doc_ll = []
    for i, doc in enumerate(heldout):
        word_ll = []
        for word in doc:
            word_ll.append(word[1] * np.log(theta[i] @ beta[:, word[0]]))
        per_word_ll = np.sum(word_ll) / np.sum(np.array(doc)[:, 1])
        doc_ll.append(per_word_ll)

    return np.mean(doc_ll)


def train_models(beta_train_corpus, theta_train_corpus, K, model, settings):
    settings["dim"]["K"] = K
    # extract covariates corresponding to the training corpus
    # settings["covariates"]["betaindex"] = settings["covariates"]["betaindex"][:len(beta_train_corpus)]
    settings["covariates"]["X"] = settings["covariates"]["X"][: len(beta_train_corpus)]
    # initialize dictionaries for different corpora
    model_beta_dictionary = Dictionary.from_corpus(beta_train_corpus)
    model_theta_dictionary = Dictionary.from_corpus(theta_train_corpus)
    # initialize models for theta and beta
    model_beta = STM(
        documents=beta_train_corpus,
        dictionary=model_beta_dictionary,
        settings=settings,
        model=model,
        init="spectral",
    )
    model_theta = STM(
        documents=theta_train_corpus,
        dictionary=model_theta_dictionary,
        settings=settings,
        model=model,
        init="spectral",
    )
    # take beta from model trained on train + test set
    model_beta.expectation_maximization(saving=False)
    # take theta from model trained on train + half of test documents
    model_theta.expectation_maximization(saving=False)

    return np.array(model_beta.beta), np.array(model_theta.theta)


def heldout_on_test(corpus, K, model, settings):

    train, test, validate = split_corpus(corpus, proportion=0.8)
    test_1, test_2 = cut_in_half(test)
    validate_1, validate_2 = cut_in_half(validate)

    beta_train_corpus = np.concatenate((train, test))
    theta_train_corpus = np.concatenate((train, test_1))

    beta, theta = train_models(
        beta_train_corpus, theta_train_corpus, settings=settings, model=model, K=K
    )

    doc_ll, expected_ll = eval_heldout(heldout=test_2, beta=beta, theta=theta)

    return doc_ll, expected_ll


def find_k(K_candidates, models, corpus, settings):
    results = [[]] * len(models)
    for i, model in enumerate(models):
        for j, K in enumerate(K_candidates):
            _, expected_ll = heldout_on_test(corpus=corpus, K=K, model=model, settings=settings)
            results[i].append(expected_ll)

    return results