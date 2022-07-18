
import json
import os
import pickle
import logging

import numpy as np
from gensim.corpora.dictionary import Dictionary
from stm import STM
from heldout import eval_heldout

logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename='logfiles/train_models.log',
    encoding='utf-8',
    level=logging.INFO)

# set input directory
ARTIFACTS_ROOT_DIR = "artifacts"

# specify model parameters
kappa_interactions = False
lda_beta = True
beta_index = None
max_em_iter = 20
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

        # input paths
        train_docs_path = os.path.join(corpus_dir_path, "train_docs.pickle")
        test_docs_path = os.path.join(corpus_dir_path, "test_docs.pickle")
        test_1_docs_path = os.path.join(corpus_dir_path, "test_1_docs.pickle")
        test_2_docs_path = os.path.join(corpus_dir_path, "test_2_docs.pickle")
        metadata_path = os.path.join(corpus_dir_path,"metadata.npy")
        #load from input directory
        with open(f"{train_docs_path}", 'rb') as f:
            train_corpus = pickle.load(f)
        with open(f"{test_docs_path}", 'rb') as f:
            test_corpus = pickle.load(f)
        with open(f"{test_1_docs_path}", 'rb') as f:
            test_1_corpus = pickle.load(f)
        with open(f"{test_2_docs_path}", 'rb') as f:
            test_2_corpus = pickle.load(f)
        
        # extract covariates corresponding to the training corpus
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
            logging.info(f'Fitting {model_type} for K={K} and Gamma-Factor {gamma_factor}...')
            model_beta.expectation_maximization(saving=True, output_dir=corpus_dir_path)
            model_theta.expectation_maximization(saving=True, output_dir=corpus_dir_path)

            # Save Likelihood
            logging.info(f'Evaluate the heldout likelihood on the remaining words...')
            heldout_llh = eval_heldout(test_2_corpus, theta=model_theta.theta, beta=model_beta.beta)
            
            model_type_path = os.path.join(corpus_dir_path,model_type)
            os.makedirs(model_type_path, exist_ok=True)
            
            heldout_path = os.path.join(corpus_dir_path,model_type, 'heldout')
            config_path = os.path.join(corpus_dir_path,model_type, 'config')
            logging.info(f'Saving into {model_type_path}.')
            np.save(heldout_path, np.array(heldout_llh))
            with open(config_path, "w") as f:
                json.dump(stm_config, f)
            
