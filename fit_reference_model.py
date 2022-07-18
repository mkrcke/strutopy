import json
import os
import numpy as np
import pandas as pd
from gensim import corpora
from stm import STM
import logging

logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename='logfiles/fit_reference_model.log',
    encoding='utf-8',
    level=logging.INFO)

# specify root directory
ARTIFACTS_ROOT_DIR = "artifacts"

# load reference corpus & corresponding dictionary
data = pd.read_csv(f"{ARTIFACTS_ROOT_DIR}/wiki_data/corpus_preproc.csv")
corpus = corpora.MmCorpus(f"{ARTIFACTS_ROOT_DIR}/wiki_data/BoW_corpus.mm")
dictionary = corpora.Dictionary.load(f"{ARTIFACTS_ROOT_DIR}/wiki_data/dictionary.mm")
# specify metadata as topical prevalence covariates
xmat = np.array(data.loc[:, ["statistics", "ml"]])

SEED = 12345 #random seed
np.random.seed(SEED)

# fit the model for K = 10,...,100 for a fixed seed (with spectral initialisation)
# and save it to artifacts/reference_model/K

for K in [10,20,30,40,50,60,70,80,90]:

    output_dir = f"{ARTIFACTS_ROOT_DIR}/reference_model/{K}"

    os.makedirs(output_dir, exist_ok=False)
    logging.info(f"Fit STM on the reference corpus assuming {K} topics")
    kappa_interactions = False # no topical content
    lda_beta = True # no topical content
    beta_index = None # no topical content
    max_em_iter = 25 # maximum number of iterations for the EM-algorithm
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

    logging.info(f'Save model to {output_dir}/stm_config.json')
    stm_config_path = os.path.join(output_dir, "stm_config.json")

    # Bookkeep corpus settings if input data changes
    stm_config.update(
        {
            "length_dictionary": len(dictionary),
            "number_of_docs": len(corpus),
        }
    )

    with open(stm_config_path, "w") as f:
        json.dump(stm_config, f)