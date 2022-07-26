# %%
import json
import logging
import os
import pickle

import numpy as np
from joblib import Parallel, delayed

from generate_docs import CorpusCreation
from utils import chunkIt

# %%
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    filename="logfiles/create_synthetic_corpora.log",
    encoding="utf-8",
    level=logging.INFO,
)


# specify root directory
ARTIFACTS_ROOT_DIR = "artifacts"

# specify input directory
reference_model_path = f"{ARTIFACTS_ROOT_DIR}/reference_model"

# specify list of K (topics)
# list_of_k_values = os.listdir(reference_model_path)
# list_of_k_values = list(map(int, list_of_k_values))

# specify global corpus settings
DGP = "STM"  # use the data generating process defined for STM (Roberts et al. 2016b)
TOTAL_N_CORPORA = 20  # in total, we want to create TOTAL_N_CORPORA for each setting
N_WORDS = 150  # each document contains 200 words
N_DOCS = 1500  # create 2000 documents per setting
LEVEL = 1  # count of boolean covariates generated for each document
TRAIN_TEST_PROPORTION = 0.8  # use 80% as training, 20% for evaluation
LIST_OF_GAMMA_FACTORS = [1,5,10]  # four choices for the gamma factor
ALPHA = "symmetric"  # dirichlet prior, only if DGP = 'LDA'
TREATMENT = False  # no specific treatment effect
SEED = 12345  # random seed

np.random.seed(SEED)
# %%

# for each combination of K and gamma factors, create 50 synthetic datasets
def create_synthetic_df(K):
    for gamma_factor in LIST_OF_GAMMA_FACTORS:

        input_dir = os.path.join(reference_model_path, str(K))

        with open(os.path.join(input_dir, "stm_config.json")) as f:
            stm_config = json.load(f)

        # read input parameters from reference corpus
        V = stm_config.get("length_dictionary")
        beta = np.load(os.path.join(input_dir, "beta_hat.npy"))
        gamma = np.load(os.path.join(input_dir, "gamma_hat.npy")) * gamma_factor

        logging.info(f"Setting: K = {K} and Gamma Factor = {gamma_factor}")
        for n in range(TOTAL_N_CORPORA):

            # specify settings-specific output directory
            output_dir = f"{ARTIFACTS_ROOT_DIR}/corpus/K_{K}_gamma_factor_{int(gamma_factor)}/corpus_{n}"

            # #check if file exists
            # if os.path.exists(output_dir):
            #     logging.info('Corpus already exists. Skip creation.')
            #     continue

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

            Corpus = CorpusCreation(beta=beta, gamma=gamma, **input_config)

            Corpus.generate_documents(remove_terms=False)
            logging.info(f"Created corpus {n}.")

            Corpus.split_corpus(proportion=TRAIN_TEST_PROPORTION)
            logging.info(f"Split corpus {n} into train/test.")

            logging.info(f"Saving corpus {n} to {output_dir}.")

            input_config_path = os.path.join(output_dir, "input_config.json")
            dictionary_path = os.path.join(output_dir, "dictionary.mm")
            documents_path = os.path.join(output_dir, "documents.pickle")
            train_docs_path = os.path.join(output_dir, "train_docs.pickle")
            test_docs_path = os.path.join(output_dir, "test_docs.pickle")
            validation_docs_path = os.path.join(output_dir, "validation_docs.pickle")
            test_1_docs_path = os.path.join(output_dir, "test_1_docs.pickle")
            test_2_docs_path = os.path.join(output_dir, "test_2_docs.pickle")
            logging.info(f"Saving metadata for corpus {n} to {output_dir}.")
            metadata_path = os.path.join(output_dir, "metadata")
            theta_path = os.path.join(output_dir,"theta_true")
            with open(input_config_path, "w") as f:
                json.dump(input_config, f)

            with open(documents_path, "wb") as f:
                pickle.dump(Corpus.documents, f)

            with open(train_docs_path, "wb") as f:
                pickle.dump(Corpus.train_docs, f)

            with open(test_docs_path, "wb") as f:
                pickle.dump(Corpus.test_docs, f)

            with open(test_1_docs_path, "wb") as f:
                pickle.dump(Corpus.test_1_docs, f)

            with open(test_2_docs_path, "wb") as f:
                pickle.dump(Corpus.test_2_docs, f)

            np.save(metadata_path, Corpus.metadata)
            np.save(theta_path, Corpus.theta)


# %%
# topics
t = [10,30,50,70]
cores_to_use = 8
# split according to maximal cores_to_use
t_split = chunkIt(t, float(len(t) / cores_to_use))

# %%
for ll in range(len(t_split)):
    with Parallel(n_jobs=len(t_split[ll]), verbose=51) as parallel:
        parallel(delayed(create_synthetic_df)(K=k) for k in t_split[ll])
