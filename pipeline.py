# %%

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.corpora.dictionary import Dictionary

from generate_docs import CorpusCreation
from heldout import find_k
from stm import STM

ARTIFACTS_ROOT_DIR = "artifacts"
LEVEL = 2

# %%

# load corpus & dictionary
data = pd.read_csv(f"{ARTIFACTS_ROOT_DIR}/wiki_data/corpus_preproc.csv")
corpus = corpora.MmCorpus(f"{ARTIFACTS_ROOT_DIR}/wiki_data/BoW_corpus.mm")
dictionary = corpora.Dictionary.load(f"{ARTIFACTS_ROOT_DIR}/wiki_data/dictionary.mm")

# set topical prevalence
xmat = np.array(data.loc[:, ["statistics", "ml"]])


for K in [10, 20, 30, 40]:

    output_dir = f"{ARTIFACTS_ROOT_DIR}/reference_model/{K}"

    os.makedirs(output_dir, exist_ok=True)

    kappa_interactions = False
    lda_beta = True
    beta_index = None
    max_em_iter = 6
    sigma_prior = 0
    convergence_threshold = 1e-5

    stm_config = {
        "content": False,
        "K": K,
        "kappa_interactions": False,
        "lda_beta": True,
        "max_em_iter": max_em_iter,
        "sigma_prior": sigma_prior,
        "convergence_threshold": convergence_threshold,
        # dtype: np.float32,
        "init_type": "random",
        # model_type="STM",
        # mode="ols",
    }

    model = STM(documents=corpus, dictionary=dictionary, X=xmat, **stm_config)
    model.expectation_maximization(saving=True, output_dir=output_dir)

    # Save stm config.
    stm_config_path = os.path.join(output_dir, "stm_config.json")

    stm_config.update(
        {
            "length_dictionary": len(dictionary),
            "number_of_docs": len(corpus),
        }
    )

    with open(stm_config_path, "w") as f:
        json.dump(stm_config, f)


# %% Generate Corpus ______________________________________________________________________________

reference_model_path = f"{ARTIFACTS_ROOT_DIR}/reference_model"

list_of_k_values = os.listdir(reference_model_path)

list_of_k_values = list(map(int, list_of_k_values))

TOTAL_N_CORPUS = 3
TRAIN_TEST_PROPORTION = 0.8
LIST_OF_GAMMA_FACTORS = [2]

for K in list_of_k_values:
    for gamma_factor in LIST_OF_GAMMA_FACTORS:

        input_dir = os.path.join(reference_model_path, str(K))

        with open(os.path.join(input_dir, "stm_config.json")) as f:
            stm_config = json.load(f)

        n_words = 100  # Hier was überlegen
        n_docs = stm_config.get("number_of_docs")
        V = stm_config.get("length_dictionary")
        beta = np.load(os.path.join(input_dir, "beta_hat.npy"))
        gamma = np.load(os.path.join(input_dir, "gamma_hat.npy")) * gamma_factor
        metadata = np.load(os.path.join(input_dir, "X.npy"))

        treatment = False
        alpha = "symmetric"
        dgp = "STM"

        for n in range(TOTAL_N_CORPUS):

            output_dir = f"{ARTIFACTS_ROOT_DIR}/corpus/K_{K}_gamma_factor_{gamma_factor}/corpus_{n}"

            os.makedirs(output_dir, exist_ok=True)

            input_config = {
                "n_docs": n_docs,
                "n_words": n_words,
                "n_topics": K,
                "V": V,
                "level": LEVEL,
                "treatment": treatment,
                "alpha": alpha,
                "dgp": dgp,
            }

            Corpus = CorpusCreation(
                beta=beta, gamma=gamma, metadata=metadata, **input_config
            )

            Corpus.generate_documents()

            Corpus.split_corpus(proportion=TRAIN_TEST_PROPORTION)

            # Save output.
            input_config_path = os.path.join(output_dir, "input_config.json")
            dictionary_path = os.path.join(output_dir, "dictionary.mm")
            documents_path = os.path.join(output_dir, "documents.pickle")
            train_docs_path = os.path.join(output_dir, "train_docs.pickle")
            test_docs_path = os.path.join(output_dir, "test_docs.pickle")
            validation_docs_path = os.path.join(output_dir, "validation_docs.pickle")
            test_1_docs_path = os.path.join(output_dir, "test_1_docs.pickle")
            test_2_docs_path = os.path.join(output_dir, "test_2_docs.pickle")

            Corpus.dictionary.save(dictionary_path)

            with open(input_config_path, "w") as f:
                json.dump(input_config, f)

            with open(documents_path, "wb") as f:
                pickle.dump(Corpus.documents, f)

            with open(train_docs_path, "wb") as f:
                pickle.dump(Corpus.train_docs, f)

            with open(test_docs_path, "wb") as f:
                pickle.dump(Corpus.test_docs, f)

            with open(validation_docs_path, "wb") as f:
                pickle.dump(Corpus.validation_docs, f)

            with open(test_1_docs_path, "wb") as f:
                pickle.dump(Corpus.test_1_docs, f)

            with open(test_2_docs_path, "wb") as f:
                pickle.dump(Corpus.test_2_docs, f)


# %%


kappa_interactions = False
lda_beta = True
beta_index = None
max_em_iter = 6
sigma_prior = 0
convergence_threshold = 1e-5

# X = np load X vermutlich reference model oder neu abspeichern beim Step davor

# take beta from model trained on train + test set
# beta_train_corpus = pickle load train, test + np.concatenate((train, test))

# Get K from folder or input_config.json

corpus_artifact_path = f"{ARTIFACTS_ROOT_DIR}/corpus"

for k_gamma_combination in os.listdir(corpus_artifact_path):
    for corpus_dir in os.listdir(
        os.path.join(corpus_artifact_path, k_gamma_combination)
    ):
        corpus_dir_path = os.path.join(
            corpus_artifact_path, k_gamma_combination, corpus_dir
        )
        print(corpus_dir_path)

        K = int(corpus_dir_path.split("_")[1])
        gamma_factor = int(corpus_dir_path.split("_")[-1])

        for model_type in ["STM", "CTM"]:

            stm_config = {
                "model_type": model_type,
                "content": False,
                "K": K,
                "kappa_interactions": False,
                "lda_beta": True,
                "max_em_iter": max_em_iter,
                "sigma_prior": sigma_prior,
                "convergence_threshold": convergence_threshold,
                "init_type": "random",
            }

            # extract covariates corresponding to the training corpus
            X_train = X[: len(beta_train_corpus)]

            # initialize dictionaries for different corpora
            model_beta_dictionary = Dictionary.from_corpus(beta_train_corpus)

            # initialize models for theta and beta
            model = STM(
                documents=beta_train_corpus,
                dictionary=model_beta_dictionary,
                X=xmat,
                **stm_config,
            )

            model.expectation_maximization(saving=True, output_dir=output_dir)

            # Save beta
