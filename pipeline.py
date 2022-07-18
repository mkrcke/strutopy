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
from heldout import find_k, eval_heldout
from stm import STM

ARTIFACTS_ROOT_DIR = "artifacts"
LEVEL = 2
N_WORDS = 200
N_DOCS = 2000

# %%

# load corpus & dictionary
data = pd.read_csv(f"{ARTIFACTS_ROOT_DIR}/wiki_data/corpus_preproc.csv")
corpus = corpora.MmCorpus(f"{ARTIFACTS_ROOT_DIR}/wiki_data/BoW_corpus.mm")
dictionary = corpora.Dictionary.load(f"{ARTIFACTS_ROOT_DIR}/wiki_data/dictionary.mm")

# set topical prevalence
xmat = np.array(data.loc[:, ["statistics", "ml"]])


for K in [10,20,30,40,50,60,70,80,90]:

    output_dir = f"{ARTIFACTS_ROOT_DIR}/reference_model/{K}"

    os.makedirs(output_dir, exist_ok=True)

    kappa_interactions = False # no topical content
    lda_beta = True # no topical content
    beta_index = None # no topical content
    max_em_iter = 20 # maximum number of iterations (before the model converges)
    sigma_prior = 0 # prior on sigma, for update of the global covariance matrix
    convergence_threshold = 1e-5 # convergence treshold, in accordance to Roberts et al. 

    stm_config = {
        "init_type": "random",
        "model_type":"STM",
        "K": K,
        "convergence_threshold": convergence_threshold,
        "lda_beta": True,
        "max_em_iter": max_em_iter,
        "kappa_interactions": False,
        "sigma_prior": sigma_prior,
        "content": False,
        # dtype: np.float32,
        # mode="ols",
    }

    # fit STM 10 times on the reference corpora with the settings specified above
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
TREATMENT = False
ALPHA = "symmetric"
DGP = "STM"


for K in list_of_k_values:
    for gamma_factor in LIST_OF_GAMMA_FACTORS:

        input_dir = os.path.join(reference_model_path, str(K))

        with open(os.path.join(input_dir, "stm_config.json")) as f:
            stm_config = json.load(f)

        #n_docs = stm_config.get("number_of_docs")
        V = stm_config.get("length_dictionary")
        beta = np.load(os.path.join(input_dir, "beta_hat.npy"))
        gamma = np.load(os.path.join(input_dir, "gamma_hat.npy")) * gamma_factor
        # metadata = np.load(os.path.join(input_dir, "X.npy")) # won't use, since doc_length does not match


        for n in range(TOTAL_N_CORPUS):

            output_dir = f"{ARTIFACTS_ROOT_DIR}/corpus/K_{K}_gamma_factor_{gamma_factor}/corpus_{n}"

            os.makedirs(output_dir, exist_ok=True)

            input_config = {
                "n_docs": N_DOCS,
                "n_words": N_WORDS,
                "n_topics": K,
                "V": V,
                "level": LEVEL,
                "treatment": TREATMENT,
                "alpha": ALPHA,
                "dgp": DGP,
            }

            Corpus = CorpusCreation(
                beta=beta, gamma=gamma, **input_config
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
            metadata_path = os.path.join(output_dir, "metadata")


            
            with open(input_config_path, "w") as f:
                json.dump(input_config, f)

            with open(documents_path, "wb") as f:
                pickle.dump(Corpus.documents, f)

            with open(train_docs_path, "wb") as f:
                pickle.dump(Corpus.train_docs, f)

            with open(test_docs_path, "wb") as f:
                pickle.dump(Corpus.test_docs, f)

            # with open(validation_docs_path, "wb") as f:
            #     pickle.dump(Corpus.validation_docs, f)

            with open(test_1_docs_path, "wb") as f:
                pickle.dump(Corpus.test_1_docs, f)

            with open(test_2_docs_path, "wb") as f:
                pickle.dump(Corpus.test_2_docs, f)

            np.save(metadata_path, Corpus.metadata)

# %% Train and evaluate models________________________________________________________________________


kappa_interactions = False
lda_beta = True
beta_index = None
max_em_iter = 2
sigma_prior = 0
convergence_threshold = 1e-5

corpus_artifact_path = f"{ARTIFACTS_ROOT_DIR}/corpus"

for k_gamma_combination in os.listdir(corpus_artifact_path):
    for corpus_dir in os.listdir(
        os.path.join(corpus_artifact_path, k_gamma_combination)
    ):
        corpus_dir_path = os.path.join(
            corpus_artifact_path, k_gamma_combination, corpus_dir
        )

        K = int(corpus_dir_path.split("_")[1])
        gamma_factor = int(corpus_dir_path.split("_")[-2].split("/")[0])

        #locations
        train_docs_path = os.path.join(corpus_dir_path, "train_docs.pickle")
        test_docs_path = os.path.join(corpus_dir_path, "test_docs.pickle")
        test_1_docs_path = os.path.join(corpus_dir_path, "test_1_docs.pickle")
        test_2_docs_path = os.path.join(corpus_dir_path, "test_2_docs.pickle")
        metadata_path = os.path.join(corpus_dir_path,"metadata.npy")
        #load from location
        with open(f"{train_docs_path}", 'rb') as f:
            train_corpus = pickle.load(f)
        with open(f"{test_docs_path}", 'rb') as f:
            test_corpus = pickle.load(f)
        with open(f"{test_1_docs_path}", 'rb') as f:
            test_1_corpus = pickle.load(f)
        with open(f"{test_2_docs_path}", 'rb') as f:
            test_2_corpus = pickle.load(f)
        
        X = np.load(metadata_path)
        
        # Prepare corpora for model training
        beta_train_corpus = [*train_corpus, *test_corpus] # unpack both iterables in a list literal
        theta_train_corpus = [*train_corpus, *test_1_corpus] # unpack both iterables in a list literal
        heldout_corpus = test_2_corpus
          
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
                "init_type": "spectral",
                }

            # extract covariates corresponding to the training corpus

            # initialize dictionaries for different corpora
            model_beta_dictionary = Dictionary.from_corpus(beta_train_corpus)
            model_theta_dictionary = Dictionary.from_corpus(theta_train_corpus)
            # initialize models for theta and beta
            model_beta = STM(
                documents=beta_train_corpus,
                dictionary=model_beta_dictionary,
                X=X[: len(beta_train_corpus)],
                **stm_config,
            )
            model_theta=STM(
                documents=theta_train_corpus,
                dictionary=model_theta_dictionary,
                X=X[: len(theta_train_corpus)],
                **stm_config,
            )
            
            # Train model to retrieve beta and theta (document completion approach)
            print(f'Fitting {model_type} for K={K} and Gamma-Factor {gamma_factor}...')
            model_beta.expectation_maximization(saving=True, output_dir=corpus_dir_path)
            model_theta.expectation_maximization(saving=True, output_dir=corpus_dir_path)

            # Save Likelihood
            print(f'Evaluate the heldout likelihood on the remaining words...')
            heldout_llh = eval_heldout(test_2_corpus, theta=model_theta.theta, beta=model_beta.beta)
            
            model_type_path = os.path.join(corpus_dir_path,model_type)
            os.makedirs(model_type_path, exist_ok=True)
            
            heldout_path = os.path.join(corpus_dir_path,model_type, 'heldout')
            config_path = os.path.join(corpus_dir_path,model_type, 'config')
            print(f'Saving into {model_type_path}.')
            np.save(heldout_path, np.array(heldout_llh))
            with open(config_path, "w") as f:
                json.dump(stm_config, f)
            

# %% Evaluation

# for each combination of K and Gamma: 
# for each model in ["STM", "CTM"]
# for each corpus in 1:50: 
# np.load(heldout_likelihood) and append to a list 

for k_gamma_combination in os.listdir(corpus_artifact_path):
    for corpus_dir in os.listdir(
        os.path.join(corpus_artifact_path, k_gamma_combination)
    ):
    print(corpus_dir)
    for model in ["STM","CTM"]