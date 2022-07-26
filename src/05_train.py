# %%
import json
import logging
import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from joblib import Parallel, delayed

from modules.chunk_it import chunkIt
from modules.heldout import eval_heldout
from modules.stm import STM

# %%
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    filename="logfiles/train_models.log",
    encoding="utf-8",
    level=logging.INFO,
)

logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)

# set input directory
ARTIFACTS_ROOT_DIR = "artifacts"

# specify model parameters
kappa_interactions = False
lda_beta = True
beta_index = None
max_em_iter = 10
sigma_prior = 0
convergence_threshold = 1e-5
corpus_artifact_path = f"{ARTIFACTS_ROOT_DIR}/corpus"

# %%


def train_on_synthetic_data(input):

    K = int(input[1].split("_")[1])
    gamma_factor = int(input[1].split("_")[-1])
    corpus = input[-1]
    model = input[-2]

    print("K:", K)
    print("gamma_factor:", gamma_factor)

    root_path = os.path.join(corpus_artifact_path, input[1], f"corpus_{str(corpus)}")
    # input paths
    train_docs_path = os.path.join(root_path, "train_docs.pickle")
    test_docs_path = os.path.join(root_path, "test_docs.pickle")
    test_1_docs_path = os.path.join(root_path, "test_1_docs.pickle")
    test_2_docs_path = os.path.join(root_path, "test_2_docs.pickle")
    metadata_path = os.path.join(root_path, "metadata.npy")
    # load from input directory
    with open(f"{train_docs_path}", "rb") as f:
        train_corpus = pickle.load(f)
    with open(f"{test_docs_path}", "rb") as f:
        test_corpus = pickle.load(f)
    with open(f"{test_1_docs_path}", "rb") as f:
        test_1_corpus = pickle.load(f)
    with open(f"{test_2_docs_path}", "rb") as f:
        test_2_corpus = pickle.load(f)

    # extract covariates corresponding to the training corpus
    X = np.load(metadata_path)

    # Prepare corpora for model training
    beta_train_corpus = [
        *train_corpus,
        *test_corpus,
    ]  # unpack both iterables in a list literal
    theta_train_corpus = [
        *train_corpus,
        *test_1_corpus,
    ]  # unpack both iterables in a list literal
    heldout_corpus = test_2_corpus

    stm_config = {
        "model_type": model,
        "content": False,
        "K": K,
        "kappa_interactions": False,
        "lda_beta": True,
        "max_em_iter": max_em_iter,
        "sigma_prior": sigma_prior,
        "convergence_threshold": convergence_threshold,
        "init_type": "spectral",
    }

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

    model_theta = STM(
        documents=theta_train_corpus,
        dictionary=model_theta_dictionary,
        X=X[: len(theta_train_corpus)],
        **stm_config,
    )

    # Train model to retrieve beta and theta (document completion approach)
    logger.info(f"Fitting {model} for K={K} and Gamma-Factor {gamma_factor}...")
    model_beta.expectation_maximization(saving=True, output_dir=root_path)
    model_theta.expectation_maximization(saving=True, output_dir=root_path)

    # Save Likelihood
    logger.info(f"Evaluate the heldout likelihood on the remaining words...")
    heldout_llh = eval_heldout(
        heldout_corpus, theta=model_theta.theta, beta=model_beta.beta
    )

    model_type_path = os.path.join(root_path, model)
    os.makedirs(model_type_path, exist_ok=True)

    heldout_path = os.path.join(root_path, model, "heldout")
    config_path = os.path.join(root_path, model, "config")
    logger.info(f"Saving into {model_type_path}.")
    np.save(heldout_path, np.array(heldout_llh))
    with open(config_path, "w") as f:
        json.dump(stm_config, f)


# %%
# reduce the loop to one layer of parallelization
# for each training, define the input in a sequence (list) and pass it to the function

k_gamma = [
    #'K_70_gamma_factor_3',
    #'K_30_gamma_factor_0',
    #'K_50_gamma_factor_0',
    #'K_10_gamma_factor_3',
    #'K_70_gamma_factor_0',
    #'K_30_gamma_factor_3',
    #'K_50_gamma_factor_3',
    #'K_10_gamma_factor_0',
    "K_70_gamma_factor_1",
    "K_30_gamma_factor_1",
    "K_50_gamma_factor_1",
    "K_10_gamma_factor_1",
    "K_70_gamma_factor_5",
    "K_30_gamma_factor_5",
    "K_50_gamma_factor_5",
    "K_10_gamma_factor_5",
    "K_70_gamma_factor_10",
    "K_30_gamma_factor_10",
    "K_50_gamma_factor_10",
    "K_10_gamma_factor_10",
]

list_input = pd.DataFrame(
    {
        "k_gamma": [k_gamma] * 20,  # Create pandas DataFrame
        "model": [["STM", "CTM"]] * 20,
        "corpus": list(range(0, 20)),
    }
)

input_array = np.array(list_input.explode("model").explode("k_gamma").reset_index())

# %% parallelize
t = input_array
cores_to_use = 8
# split according to maximal cores_to_use
t_split = chunkIt(t, float(len(t) / cores_to_use))
Parallel(n_jobs=len(t_split))(
    delayed(train_on_synthetic_data)(input) for input in input_array
)
