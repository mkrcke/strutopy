# %%
import json
import logging
import math
import os
import pickle
import time
import warnings
from operator import itemgetter

import numpy as np
import numpy.random as random
import pandas as pd
import scipy as sp
import sklearn.linear_model
from pandas import Series
from pyexpat import model
from qpsolvers import solve_qp
from scipy import optimize
from scipy.sparse import csr_array, csr_matrix, diags
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, normalize

# custom packages
from .generate_docs import CorpusCreation

logger = logging.getLogger(__name__)


def spectral_init(corpus, K, V, maxV=5000, verbose=True, print_anchor=False):
    """
    init='spectral' provides a deterministic initialization using the
    spectral algorithm given in Arora et al 2014.  See Roberts, Stewart and
    Tingley (2016) for details and a comparison of different approaches.
    The spectral initialisation is recommended if the number of documents is
    relatively large (e.g. D > 40.000). When the vocab is larger than 10000 terms
    only the most 10.000 frequent terms are used for initialisation. The maximum
    number of terms can be adjusted via the @param maxV. Note that the computation
    complexity increases on the magnitude of n^2.

    Numerical instabilities might occur (c.f. https://github.com/bstewart/stm/issues/133)

    @param: corpus in bag-of-word format -> [list of (int, int)]
    @param: K number of topics used for the spectral initialisation
    @param: (default=10000) maxV maximum number of most frequent terms used for spectral initialisation
    @param: verbose if True prints information as it progresses.
    @param: (default=False) print_anchor words from input documents

    @return: word-topic distribution obtained from spectral learning (K X V)
    """
    doc_term_matrix = create_dtm(corpus=corpus)

    wprob = np.sum(doc_term_matrix, axis=0)
    wprob = wprob / np.sum(wprob)
    wprob = np.array(wprob).flatten()

    keep = np.argsort(-1 * wprob)[:maxV]
    doc_term_matrix = doc_term_matrix[:, keep]
    wprob = wprob[keep]

    if verbose:
        print("Create gram matrix...")
    Q = gram(doc_term_matrix)

    if verbose:
        print("Find anchor words...")

    anchor = fastAnchor(Q, K)
    if print_anchor:
        for i, idx in enumerate(anchor):
            print(
                f"{i}. anchor word: {vectorizer.get_feature_names_out()[np.int0(idx)]}"
            )
    if verbose:
        print("Recover values for beta")
    beta = recover_l2(Q, anchor, wprob)

    if keep is not None:
        beta_new = np.zeros(K * V).reshape(K, V)
        beta_new[:, keep] = beta
        beta_new = beta_new + 0.001 / V
        beta = beta_new / np.sum(beta_new)

    return beta


def create_dtm(corpus):
    """
    Create a sparse csr_matrix constructed from three arrays:

    - data[:] word counts: the entries of the matrix, in any order
    - i[:] document index: the row indices of the matrix entries
    - j[:] word index: the column indices of the matrix entries

    @param: corpus in bag-of-word format -> [list of (int, int)]
        corpus (list): list containing word indices and word counts per document

    @return: doc_term_matrix sparse matrix with document-term counts in csr format
    """

    word_idx = []
    for doc in corpus:
        for word in doc:
            word_idx.append(word[0])
    word_idx = np.array(word_idx)

    doc_idx = []
    for i, doc in enumerate(corpus):
        for word in doc:
            doc_idx.append(i)
    doc_idx = np.array(doc_idx)

    count = []
    for doc in corpus:
        for word in doc:
            count.append(word[1])
    count = np.array(count)

    return csr_matrix((count, (doc_idx, word_idx)))


def gram(doc_term_matrix):
    """ "
    Computes a square matrix Q from the document term matrix.
    Values of Q are row normalized.

    Note: scipy.sparse matrix features are used to improve computation time

    @param: doc_term_matrix (D x V) in sparse csr format

    @return: sparse row-normalized matrix Q (VxV) in sparse csr format
    """
    # absolute word counts per document
    word_counts = doc_term_matrix.sum(axis=1)

    # TODO: remove terms occuring less than twice
    # doc_term_matrix= doc_term_matrix[word_counts>=2,]
    # word_counts = word_counts[word_counts>=2]

    # convert to dense arrays for faster computation
    divisor = np.array(word_counts) * np.array(word_counts - 1)

    # convert back to sparse matrices to save some time
    doc_term_matrix = csr_matrix(doc_term_matrix)
    Htilde = csr_matrix(doc_term_matrix / np.sqrt(divisor))
    Hhat = diags(np.array(np.sum(doc_term_matrix / divisor, axis=0)).flatten(), 0)

    # compute Q matrix (takes some time)
    Q = Htilde.T @ Htilde - Hhat

    # normalize Q:
    assert np.all(
        Q.sum(axis=1) > 0
    ), "Encountered zeroes in Q row sums, can not normalize."
    # row-normalise Q
    normalize(Q, copy=False)
    return Q


def fastAnchor(Q, K, verbose=True):
    """Find Anchor Words

    Take matrix Q and return an anchor term for each of K topics.
    Projection of all words onto the basis spanned by the anchors.

    Note: scipy.sparse matrix features are used to improve computation time

    @input: Q The gram matrix
    @input: K The number of desired anchors
    @input: verbose If True prints a dot to the screen after each anchor

    @return: anchor vector of indices for rows of Q containing anchors
    """
    # compute squared sum per row using scipy.sparse
    row_squared_sum = csr_array(Q.power(2).sum(axis=0))
    basis = np.zeros(K)

    for i in range(K):
        # most probable word over topics
        maxind = row_squared_sum.argmax()
        basis[i] = maxind
        maxval = row_squared_sum.max()
        normalizer = 1 / np.sqrt(maxval)

        # normalize the high probable word (maxind)
        Q[maxind] = Q[maxind] * normalizer

        # For each row
        innerproducts = (
            Q
            @ Q[
                maxind,
            ].T
        )

        # Each row gets multiplied out through the Basis
        # (use numpy array as we are not gaining anything for sparse vectors)
        if i == 0:
            project = (
                innerproducts.toarray()
                @ Q[
                    maxind,
                ].toarray()
            )

        project = (
            innerproducts
            @ Q[
                maxind,
            ]
        )

        # Now we want to subtract off the projection but
        # first we should zero out the components for the basis
        # vectors which we weren't intended to calculate
        project[
            np.int0(basis),
        ] = 0
        Q = Q.A - project

        # Q is not sparse anymore...
        row_squared_sum = np.sum(np.power(Q, 2), axis=0)
        row_squared_sum[:, np.int0(basis)] = 0
        if verbose:
            print(".", end="", flush=True)
    return basis


def recover_l2(Q, anchor, wprob):
    """
    Recover the topic-word parameters from a set of anchor words using the RecoverL2
    procedure of Arora et al.

    Using the exponentiated algorithm and an L2 loss identify the optimal convex
    combination of the anchor words which can reconstruct each additional word in the
    matrix. Transform and return as a beta matrix.

    @param: Q the row-normalized gram matrix
    @param: anchor A vector of indices for rows of Q containing anchors
    @param: wprob The empirical word probabilities used to renorm the mixture weights.

    @return: A (np.ndarray): word-topic matrix of dimension K by V
    """
    # Prepare Quadratic Programming
    M = Q[
        np.int0(anchor),
    ]
    P = np.array(np.dot(M, M.T).todense())  # square matrix

    # # condition Ax=b coefficients sum to 1
    # A = np.ones(M.shape[0])
    # b = np.array([1])

    # conditino Gx >= h (-Gx >= 0 <=> Gx <= 0) coefficients greater or equal to zero
    G = np.eye(M.shape[0], M.shape[0])
    h = np.zeros(M.shape[0])

    # initialize empty array to store solutions
    condprob = np.empty(Q.shape[0], dtype=np.ndarray, order="C")
    # find convex solution for each word seperately:
    for i in range(Q.shape[0]):
        if i in anchor:
            vec = np.repeat(0, P.shape[0])
            vec[np.where(anchor == i)] = 1
            condprob[i] = vec
        else:
            y = Q[
                i,
            ]
            q = np.array((M @ y.T).todense()).flatten()
            solution = solve_qp(
                P=P,
                q=q,
                G=G,
                h=h,
                verbose=True,
                # lower/upper bound for probabilities
                # lb = np.zeros(M.shape[0]),
                # ub = np.ones(M.shape[0]),
                solver="quadprog",
            )
            # replace small negative values with epsilon and store solution
            # if np.any(solution<0):
            #     solution[solution<0] = np.finfo(float).eps
            condprob[i] = -1 * solution

    # p(z|w)
    weights = np.vstack(condprob)
    # p(w|z) = p(z|w)p(w)
    A = weights.T * wprob
    # transform
    A = A.T / np.sum(A, axis=1)
    # check probability assumption
    assert np.any(A > 0), "Negative probabilities for some words."
    assert np.any(A < 1), "Word probabilities larger than one."
    return A.T


############ TESTING ######################
# data = pd.read_csv('data/poliblogs2008.csv')
# # selection for quick testing (make sure it is in line with R selection)
# data = data[:100]
# # use the count vectorizer to get absolute word counts
# vectorizer = CountVectorizer()
# doc_term_matrix = vectorizer.fit_transform(data.documents)
# K=10
# beta = spectral_init(doc_term_matrix, maxV=None, verbose=True, K=K)


class STM:
    def __init__(
        self,
        documents,
        dictionary,
        content,
        K,
        X,
        kappa_interactions,
        lda_beta,
        max_em_iter,
        sigma_prior,
        convergence_threshold,
        beta_index=None,
        A=None,
        dtype=np.float32,
        init_type="spectral",
        model_type="STM",
        mode="ols",
    ):
        """
        @param: documents BoW-formatted documents in list of list of arrays with index-count tuples for each word
                example: `[[(1,3),(3,2)],[(1,1),(4,2)]]` -> [list of (int, int)]
        @param: dictionary contains word-indices of the corpus
        @param: content {dtype=bool} specifies whether topical content model is included or not
        @param: init (default='spectral') init method to be used to initialise the word-topic distribution beta.
                One might choose between 'random' and 'spectral', however the spectral initialisation is recommended
        @param: model (default='STM') to update variational mean parameter for the topical prevalence. Choose between
                'STM' and 'CTM'. Note, however, that 'CTM' updates ignore the topical prevalence model.
        @param: mode (default='ols') to estimate the prevalence coefficients (gamma). Otherwise choose between l1 & l2 norm.
        @param: dtype (default=np.float32) used for value checking along the process

        @return:initialised values for the algorithm specifications parameters
                    - covar: topical prevalence covariates
                    # - enet: elastic-net configuration for the regularized update of the variational distribution over topics
                    - interactions: (bool) whether interactions between topics and covariates should be modelled (True) or not (False)
                    - betaindex: index for the topical prevalence covariate level (equals covar at the moment)
                    - last_bound: list to store approximated bound for each EM-iteration
                initialised values for the global
                    - V: number of tokens
                    - K: number of topics
                    - N: number of documents
                    - A: topical prevalence covariate levels
                    - mu: prior on topical prevalence (N x (K-1))
                    - sigma: prior covariance on topical prevalence ((K-1) x (K-1))
                    - beta: prior on word-topic distribution (K x V)
                    - eta: prior on document-topic distribution (N x (K-1))
                    - theta: simplex mapped version of eta representing the document-topic distribution (N x K)
                    - kappa: prior on topical content covariates
        """

        random.seed(123456)

        self.dtype = np.finfo(dtype).dtype

        # Specify Corpus & Settings
        self.documents = documents
        self.dictionary = dictionary
        self.init = init_type
        self.model = model_type
        self.mode = mode
        self.content = content

        self.K = K
        self.A = A
        self.V = len(self.dictionary)
        self.X = X
        self.interactions = kappa_interactions
        self.beta_index = beta_index
        self.max_em_iter = max_em_iter
        self.sigma_prior = sigma_prior
        self.convergence_threshold = convergence_threshold
        self.N = len(self.documents)
        self.LDAbeta = lda_beta
        self.betaindex = beta_index

        # convergence settings
        self.last_bounds = []
        self.max_em_its = max_em_iter

        # test and store user-supplied parameters
        if len(documents) is None:
            raise ValueError("documents must be specified to establish input space")
        if self.K == 0:
            raise ValueError("Number of topics must be specified")
        if self.A == 1:
            logging.warning("no dimension for the topical content provided")

        # initialise
        self.init_params()

    # _____________________________________________________________________
    def init_params(self):
        """initialises global parameters beta, mu, eta, sigma and kappa"""
        # Set global params
        self.init_beta()
        self.init_mu()
        self.init_eta()
        self.init_sigma()
        self.wcounts()
        self.init_theta()

    def init_beta(self):
        """
        beta : {float, numpy.ndarray of float, list of float, str}, optional
            A-priori belief on topic-word distribution, this can be:
                * scalar for a symmetric prior over topic-word distribution,
                * 1D array of length equal to num_words to denote an asymmetric user defined prior for each word,
                * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination.
        """
        if self.init == "spectral":
            self.beta = spectral_init(
                self.documents, self.K, self.V, maxV=5000, verbose=False
            )
        elif self.init == "random":
            beta_init = random.gamma(0.1, 1, self.V * self.K).reshape(self.K, self.V)
            row_sums = np.sum(beta_init, axis=1)[:, None]
            beta_init_normalized = np.divide(
                beta_init, row_sums, out=np.zeros_like(beta_init), where=row_sums != 0
            )
            if self.interactions:  # TODO: replace ifelse condition by logic
                self.beta = np.repeat(beta_init_normalized[None, :], self.A, axis=0)
                # test if probabilities sum to 1
                # [
                #     np.testing.assert_almost_equal(sum_over_words, 1)
                #     for i in range(self.A)
                #     for sum_over_words in np.sum(self.beta[i], axis=1)
                # ]
            else:
                self.beta = beta_init_normalized
                # [
                #     np.testing.assert_almost_equal(sum_over_words, 1)
                #     for i in range(self.A)
                #     for sum_over_words in np.sum(self.beta, axis=1)
                # ]
        # assert self.beta.shape == (
        #     self.A,
        #     self.K,
        #     self.V,
        # ), "Invalid beta shape. Got shape %s, but expected (%s, %s)" % (
        #     str(self.beta.shape),
        #     str(self.K),
        #     str(self.V),
        # )

    # TODO: Check for the shape of mu if this is correct
    def init_mu(self):
        self.mu = np.zeros((self.N, self.K - 1))

    def init_sigma(self):
        self.sigma = np.zeros(((self.K - 1), (self.K - 1)))
        np.fill_diagonal(self.sigma, 20)

    def init_eta(self):
        """
        dimension: N by K-1
        """
        self.eta = np.zeros((self.N, self.K - 1))

    def init_theta(self):
        """document level parameter to store the mean topic probabilities
        based on the numerical optimization and the log additive transformation.
        (simplex mapped version of eta)

        dimension: N by K
        """
        self.theta = np.zeros((self.N, self.K))

    def init_gamma(self):
        """The prior specification for the topic prevalence parameters is a zero mean Gaussian distribution with shared variance parameter,
        gamma_p,k ~ N(0,sigma_k^2)
        sigma_k^2 ~ InverseGamma(a,b), with a & b fixed
        """
        self.gamma = np.zeros(self.level, self.K)

    def wcounts(self):
        self.wcounts = np.array(create_dtm(self.documents).sum(axis=0)).flatten()

    # _____________________________________________________________________
    def E_step(self):
        """
        updates the sufficient statistics for each e-step iteration:
            sigma_ss: np.array of shape ((k-1), (k-1))
            beta_ss: np.array of shape (k, v) but might need A
            bound: might be implemented as list len(#iterations)
        """
        # 2) Precalculate common components
        while True:
            try:
                sigobj = np.linalg.cholesky(self.sigma)
                self.sigmaentropy = np.sum(np.log(np.diag(sigobj)))
                self.siginv = np.linalg.inv(sigobj).T * np.linalg.inv(sigobj)
                break
            except:
                logging.ERROR(
                    "Cholesky Decomposition failed, because Sigma is not positive definite."
                )
                self.sigmaentropy = (
                    0.5 * np.linalg.slogdet(self.sigma)[1]
                )  # part 2 of ELBO
                self.siginv = sp.linalg.cholesky(self.sigma)  # part 3 of ELBO

        # initialize sufficient statistics
        calculated_bounds = []
        sigma_ss = np.zeros(shape=self.sigma.shape)
        beta_ss = np.zeros(shape=self.beta.shape)

        start_time = time.time()

        for i in range(self.N):

            # set document specs
            doc_array = np.array(self.documents[i])
            idx_1v = doc_array[
                :, 0
            ]  # This counts the first dimension of the numpy array, was "idx_1v"

            if self.content:
                aspect = self.betaindex[i]
            else:
                aspect = None

            beta_doc_kv = self.get_beta(idx_1v, aspect=aspect)
            word_count_1v = doc_array[:, 1]
            assert np.all(beta_doc_kv >= 0), "Some entries of beta are negative or nan."

            # optimize variational posterior
            # does not matter if we use optimize.minimize(method='BFGS') or optimize fmin_bfgs()
            res = self.optimize_eta(
                eta=self.eta[i],
                mu=self.mu[i],
                word_count=word_count_1v,
                beta_doc=beta_doc_kv,
            )

            # print(f"document {i}:", res.message)
            self.eta[i] = res.x
            self.theta[i] = np.exp(np.insert(res.x, self.K - 1, 0)) / np.sum(
                np.exp(np.insert(res.x, self.K - 1, 0))
            )
            # Compute Hessian, Phi and Lower Bound
            # 1) check if inverse is a legitimate cov matrix
            # 2) if not, adjust matrix to be positive definite
            hess_i = self.hessian(
                eta=self.eta[i], word_count=word_count_1v, beta_doc_kv=beta_doc_kv
            )
            L_i = self.decompose_hessian(hess_i, approx=res.hess_inv)

            # Delta Bound
            bound_i = self.lower_bound(
                L_i,
                mu=self.mu[i],
                word_count=word_count_1v,
                beta_doc_kv=beta_doc_kv,
                eta=self.eta[i],
            )

            # Delta NU
            nu = self.optimize_nu(L_i)

            # Delta Phi

            self.update_z(
                eta=self.eta[i],
                beta_doc_kv=beta_doc_kv,
                word_count=word_count_1v,
            )

            # print(bound_i)
            calculated_bounds.append(bound_i)

            # update sufficient statistics
            sigma_ss += nu
            # TODO: combine into one
            if self.interactions:
                beta_ss[aspect][:, np.array(np.int0(idx_1v))] += self.phi
            else:
                try:
                    beta_ss[:, np.array(np.int0(idx_1v))] += self.phi
                except RuntimeWarning:
                    breakpoint()

        self.bound = np.sum(calculated_bounds)
        self.last_bounds.append(self.bound)
        elapsed_time = np.round((time.time() - start_time), 3)
        logging.info(f"Lower Bound: {self.bound}")
        logging.info(f"Completed E-Step in {elapsed_time} seconds. \n")
        return beta_ss, sigma_ss

    def get_beta(self, words, aspect):
        """returns the topic-word distribution for a document with the respective topical content covariate (aspect)

        Args:
            words (ndarray): 1D-array with word indices for a specific document
            aspect (int, float32): topical content covariate for a specific document

        Raises:
            ValueError: _description_

        Returns:
            beta_doc_kv: topic-word distribution for a specific document, based on word indices and aspect
        """
        # if not np.all((self.beta >= 0)):
        # raise ValueError("Some entries of beta are negative.")
        if self.interactions:
            beta_doc_kv = self.beta[aspect][:, np.array(np.int0(words))]
        else:
            beta_doc_kv = self.beta[:, np.array(np.int0(words))]
        # add small value to beta for numerical stability
        # beta_doc_kv += 1e-10
        return beta_doc_kv

    def M_step(self, beta_ss, sigma_ss):
        # Run M-Step

        start_time = time.time()

        self.update_mu()

        self.update_sigma(nu=sigma_ss, sigprior=self.sigma_prior)

        self.update_beta(beta_ss)

        elapsed_time = np.round((time.time() - start_time), 3)
        logging.info(f"Completed M-Step in {elapsed_time} seconds. \n")

    def update_mu(self, intercept=True):
        """
        updates the mean parameter for the [document specific] logistic normal distribution.
        Changing estimation of prevalence covariate effects:
            - if CTM, the prevalence covariate effects are not included, hence we have the CTM specification
            - if STM, the prevalence covariates are included with one of the following options:
                - mode == "lasso": uses the l1-loss for regularizing parameters
                - mode == "ridge": uses the l2-loss for regularizing parameters
                - mode == "ols": coefficients are estimated using ordinary least squares estimation
        @param: mode (default: 'ols') for estimation of coefficients in the linear model
        @param: intercept (default: True) whether or not an intercept is included in the model
        """
        if self.model == "CTM":
            # assert self.A < 2, 'Uses column means for the mean, since no covariates are specified.'
            # just use the mean for all documents
            self.mu = np.repeat(np.mean(self.eta, axis=0)[None, :], self.N, axis=0)

        # mode = L1 simplest method requires only glmnet (https://cran.r-project.org/web/packages/glmnet/index.html)
        elif self.model == "STM":
            # prepare covariate matrix for modeling
            try:
                self.X = self.X.astype("category")
            except:
                pass

            prev_cov = np.array(self.X)[
                :, None
            ]  # prepares 1D array for one-hot encoding (OHE) by making it 2D

            # remove empty dimension
            if len(prev_cov.shape) > 2:
                prev_cov = np.squeeze(prev_cov, axis=1)

            if not np.array_equal(prev_cov, prev_cov.astype(bool)):
                enc = OneHotEncoder(handle_unknown="ignore")  # create OHE
                prev_cov = enc.fit_transform(prev_cov).toarray()  # fit OHE

            if self.mode not in ["lasso", "ridge", "ols"]:
                print(
                    "Need to specify the estimation mode of prevalence covariate coefficients. Uses default 'ols'."
                )

            if self.mode == "lasso":
                linear_model = sklearn.linear_model.Lasso(
                    alpha=1, fit_intercept=intercept
                )
                fitted_model = linear_model.fit(prev_cov, self.eta)

            elif self.mode == "ridge":
                linear_model = sklearn.linear_model.Ridge(
                    alpha=0.1, fit_intercept=intercept
                )
                fitted_model = linear_model.fit(prev_cov, self.eta)

            else:
                linear_model = sklearn.linear_model.LinearRegression(
                    fit_intercept=intercept
                )
                fitted_model = linear_model.fit(prev_cov, self.eta)

            # adjust design matrix if intercept is estimated
            if intercept:
                self.gamma = np.column_stack(
                    (fitted_model.intercept_, fitted_model.coef_)
                )
                design_matrix = np.c_[np.ones(prev_cov.shape[0]), prev_cov]

            self.gamma = fitted_model.coef_
            design_matrix = prev_cov

            self.mu = design_matrix @ self.gamma.T

        else:
            raise ValueError(
                'Updating the topical prevalence parameter requires a mode. Choose from "CTM", "Pooled" or "L1" (default).'
            )

    def update_sigma(self, nu, sigprior):
        """
        Updates the variance covariance matrix for the logistic normal distribution of topical prevalence

        Args:
            nu (_type_): variance-covariance for the variational document-topic distribution
            sigprior (_type_): prior for the var-cov. matrix for the log-normal
        """
        # find the covariance
        covariance = (self.eta - self.mu).T @ (self.eta - self.mu)
        covariance = np.array(covariance, dtype="float64")
        sigma = (covariance + nu) / self.N
        sigma = np.array(sigma, dtype="float64")
        self.sigma = np.diag(np.diag(sigma)) * sigprior + (1 - sigprior) * sigma

    def update_beta(self, beta_ss):
        """
        Updates the topic-word distribution beta

        @param: beta_ss (dtype:np.ndarray) sufficient statistic for word-topic distribution

        if self.LDAbeta == True: Row-normalization of beta for the update
        if self.LDAbeta == False: Distributed Poisson Regression for the updates
        """
        if self.LDAbeta:
            assert np.any(np.sum(beta_ss, axis=1) >= 0), "break here"
            row_sums = np.sum(beta_ss, axis=1)[:, None]
            self.beta = np.nan_to_num(self.beta)
            self.beta = np.divide(
                beta_ss, row_sums, out=np.zeros_like(beta_ss), where=row_sums != 0
            )
        else:
            self.mnreg(beta_ss=beta_ss)

    def mnreg(self, beta_ss):
        """estimation of distributed poisson regression for the update of the kappa parameters

        @param: beta_ss (np.ndarray) estimated word-topic distribution of the current EM-iteration with dimension K x V
        """

        contrast = False
        interact = True
        fixed_intercept = True
        alpha = 250  # corresponds to `lambda` in glmnet
        maxit = 1e4
        tol = 1e-5

        counts = csr_matrix(
            np.concatenate((beta_ss[0], beta_ss[1]), axis=0)
        )  # dimension (A*K) x V # TODO: enable dynamic creation of 'counts'

        # Three cases
        if self.A == 1:  # Topic Model
            covar = np.diag(np.ones(self.K))
        if self.A != 1:  # Topic-Aspect Models
            # if not contrast:
            # Topics
            veci = np.arange(0, counts.shape[0])
            vecj = np.tile(np.arange(0, self.K), self.A)
            # aspects
            veci = np.concatenate((veci, np.arange(0, (counts.shape[0]))))
            vecj = np.concatenate(
                (vecj, np.repeat(np.arange(self.K + 1, self.K + self.A + 1), self.K))
            )
            if interact:
                veci = np.concatenate((veci, np.arange(0, counts.shape[0])))
                vecj = np.concatenate(
                    (
                        vecj,
                        np.arange(
                            self.K + self.A + 1, self.K + self.A + counts.shape[0] + 1
                        ),
                    )
                )  # TODO: remove +1 at the end, make shapes fit anyway
            vecv = np.ones(len(veci))
            covar = csr_matrix((vecv, (veci, vecj)))

        if fixed_intercept:
            m = self.wcounts
            m = np.log(m) - np.log(np.sum(m))
        else:
            m = 0

        mult_nobs = counts.sum(axis=1)
        offset = np.log(mult_nobs)
        # counts = np.split(counts, counts.shape[1], axis=1)

        ############################
        ### Distributed Poissons ###
        ############################
        # TODO: Scaling the data for convergence
        out = []
        # now iterate over the vocabulary
        for i in range(counts.shape[1]):

            if np.all(m == 0):
                offset2 = offset
                fit_intercept = True
            else:
                fit_intercept = False
            offset2 = m[i] + offset
            mod = None
            # while mod is None:
            # alpha = alpha * np.floor(0.2*alpha)
            clf = sklearn.linear_model.PoissonRegressor(
                fit_intercept=fit_intercept,
                max_iter=np.int0(maxit),
                tol=tol,
                alpha=np.int0(alpha),
            )
            mod = clf.fit(covar, counts[:, [1]].A.flatten())
            # if it didn't converge, increase nlambda paths by 20%
            # if(is.null(mod)) nlambda <- nlambda + floor(.2*nlambda)
            # print(f'Estimated coefficients for word {i}.')
            # print(mod.coef_)
            out.append(mod.coef_)

        # put all regression results together
        coef = np.stack(out, axis=1)

        # seperate intercept from the coefficients
        if not fixed_intercept:
            m = coef[0]
            coef = coef[1:]

        # set kappa
        self.kappa = coef

        ###################
        ### predictions ###
        ###################

        linpred = covar @ coef
        linpred = m + linpred
        explinpred = np.exp(linpred)
        beta = explinpred / np.sum(explinpred, axis=1)[:, np.newaxis]

        # retain former structure for beta
        self.beta = np.split(beta, 2, axis=0)

    def expectation_maximization(self, saving, output_dir=None):
        first_start_time = time.time()
        logging.info(f"Fit STM for {self.K} topics")
        logging.info("––––––––––––––––––––––––––––––––––––––––––––––––––––––––")
        for _iteration in range(100):
            logging.info(f"E-Step iteration {_iteration}")
            beta_ss, sigma_ss = self.E_step()
            logging.info(f"M-Step iteration {_iteration}")
            self.M_step(beta_ss, sigma_ss)
            if self.EM_is_converged(_iteration):
                self.time_processed = time.time() - first_start_time
                logging.info(
                    f"model converged in iteration {_iteration} after {self.time_processed}s"
                )
                break
            if self.max_its_reached(_iteration):
                self.time_processed = time.time() - first_start_time
                logging.info(
                    f"maximum number of iterations ({self.max_em_its}) reached after {self.time_processed} seconds"
                )
                break

        if saving:
            logging.info(f"saving model to {output_dir}")
            assert output_dir is not None
            self.save_model(output_dir)

    # _____________________________________________________________________
    def EM_is_converged(self, _iteration, convergence=None):

        if _iteration < 1:
            return False

        new = self.bound
        old = self.last_bounds[-2]

        convergence_check = np.abs((new - old) / np.abs(old))
        logging.info(f"relative change: {convergence_check}")
        if convergence_check < self.convergence_threshold:
            return True
        else:
            return False

    def max_its_reached(self, _iteration):
        if _iteration == self.max_em_its - 1:
            return True
        else:
            _iteration += 1
            return False

    def stable_softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        xshift = x - np.max(x)
        exps = np.exp(xshift)
        return exps / np.sum(exps)

    def softmax_weights(self, x, weight):
        """Compute weighted softmax values for each sets of scores in x."""
        xshift = x - np.max(x)
        exps = weight * np.exp(xshift)[:, None]
        return exps / np.sum(exps)

    def optimize_eta(self, eta, mu, word_count, beta_doc):
        """Optimizes the variational parameter eta given the likelihood and the gradient function"""

        def f(eta, word_count, mu, beta_doc):
            """Objective for the variational update q(eta)

            Args:
                eta (_type_): mean topic distribution of document d
                word_count (_type_): count of words of document d
                beta_doc (_type_): word-topic distribution for document d

            Returns:
                _type_: function value for the objective f()
            """
            # precomputation
            eta = np.insert(eta, self.K - 1, 0)
            Ndoc = int(np.sum(word_count))

            return np.float64(
                (0.5 * (eta[:-1] - mu).T @ self.siginv @ (eta[:-1] - mu))
                - (
                    np.dot(
                        word_count,
                        eta.max() + np.log(np.exp(eta - eta.max()) @ beta_doc),
                    )
                    - Ndoc * sp.special.logsumexp(eta)
                )
            )

        def df(eta, word_count, mu, beta_doc):
            """Gradient for the objective of the variational update q(etas)"""
            eta = np.insert(eta, self.K - 1, 0)
            # formula
            return np.array(
                np.float64(
                    self.siginv @ (eta[:-1] - mu)
                    - (
                        beta_doc @ (word_count / np.sum(beta_doc.T, axis=1))
                        - (np.sum(word_count) / np.sum(np.exp(eta))) * np.exp(eta)
                    )[:-1]
                )
            )

        return optimize.minimize(
            f, x0=eta, args=(word_count, mu, beta_doc), jac=df, method="BFGS"
        )

    def make_pd(self, M):
        """
        Convert matrix X to be positive definite.

        The following are necessary (but not sufficient) conditions for a Hermitian matrix A
        (which by definition has real diagonal elements a_(ii)) to be positive definite.

        1. a_(ii)>0 for all i,
        2. a_(ii)+a_(jj)>2|R[a_(ij)]| for i!=j,
        3. The element with largest modulus lies on the main diagonal,
        4. det(A)>0.

        Returns: ValueError if matrix is not positive definite
        """
        dvec = M.diagonal()
        magnitudes = np.sum(abs(M), 1) - abs(dvec)
        # cholesky decomposition works only for symmetric and positive definite matrices
        dvec = np.where(dvec < magnitudes, magnitudes, dvec)
        # A Hermitian diagonally dominant matrix A with real non-negative diagonal entries is positive semidefinite.
        np.fill_diagonal(M, dvec)
        return M

    def hessian(self, eta, word_count, beta_doc_kv):
        """Computes the hessian matrix for the objective function.

        Args:
            eta (np.ndarray): document specific prior on topical prevalence of dimension 1 x K
            word_count (np.ndarray): document specific vector of word counts of dimension 1 x V_doc
            beta_doc_kv (np.ndarray): document specific word-topic distribution of dimension K x V_doc

        Returns:
            f: (negative) Hessian matrix as specified in Roberts et al. (2016b)
        """
        eta_ = np.insert(eta, self.K - 1, 0)
        theta = self.stable_softmax(eta_)

        a = np.transpose(np.multiply(np.transpose(beta_doc_kv), np.exp(eta_)))  # KxV
        b = np.multiply(a, np.transpose(np.sqrt(word_count))) / np.sum(a, 0)  # KxV
        c = np.multiply(b, np.transpose(np.sqrt(word_count)))  # KxV

        hess = b @ b.T - np.sum(word_count) * np.multiply(
            theta[:, None], theta[None, :]
        )
        # assert self.check_symmetric(hess), "Hessian is not symmetric"
        # broadcasting, works fine
        # difference to the c++ implementation comes from unspecified evaluation order: (+) instead of (-)
        np.fill_diagonal(
            hess, np.diag(hess) - np.sum(c, axis=1) + np.sum(word_count) * theta
        )

        d = hess[:-1, :-1]
        f = d + self.siginv

        if not np.all(np.linalg.eigvals(f) > 0):
            # print("Hessian not positive definite. Introduce Diagonal Dominance...")
            f = self.make_pd(f)
            if not np.all(np.linalg.eigvals(f) > 0):
                np.fill_diagonal(f, np.diag(f) + 1e-5)
                # print(
                #     "Hessian not positive definite.  Adding a small prior 1e-5 for numerical stability."
                # )

        return f

    def check_symmetric(self, M, rtol=1e-05, atol=1e-08):
        return np.allclose(M, np.transpose(M), rtol=rtol, atol=atol)

    def decompose_hessian(self, hess, approx):
        """
        Decompose hessian via cholesky decomposition
            - hessian needs to be symmetric and positive definite
        error -> not properly converged: make the matrix positive definite
        np.linalg.cholesky(a) requires the matrix a to be hermitian positive-definite
        """

        try:
            L = np.linalg.cholesky(hess)
        except:
            try:
                L = np.linalg.cholesky(self.make_pd(hess))
                # print("converts Hessian via diagonal-dominance")
            except:
                L = sp.linalg.cholesky(
                    self.make_pd(hess) + 1e-5 * np.eye(hess.shape[0])
                )
                # print("adds a small number to the hessian")
        return L

    def optimize_nu(self, L):
        """Given the lower triangular of the decomposition of the hessian,
        returns the variance-covariance matrix for the variational distribution: nu = inv(-hessian) = chol(-hessian) = L*L.T

        Args:
            L (np.array): lower triangular matrix of cholesky decomposition

        Returns:
            nu (np.array): variance-covariance matrix for the variational distribution q(eta|lambda, nu).
        """
        nu = np.linalg.inv(
            np.triu(L.T)
        )  # watch out: L is already a lower triangular matrix!
        nu = np.dot(nu, np.transpose(nu))
        return nu

    def lower_bound(self, L, mu, word_count, beta_doc_kv, eta):
        """_summary_

        Args:
            L (np.array): 2D_array ((K-1) x(K-1)) representing lower triangular matrix of cholesky decomposition for the variational distribution of eta
            mu (np.array): 1D-array representing mean parameter for the logistic normal distribution
            word_count (np.array): 1D-array of word counts for each document
            beta_doc_kv (np.array): 2D-array (K by V) containing the topic-word distribution for a specific document
            eta (_type_): 1D-array representing prior to the document-topic distribution

        Returns:
            bound: evidence lower bound (E(...))
            phi: update for the variational latent parameter z
        """
        eta_ = np.insert(eta, self.K - 1, 0)
        theta = self.stable_softmax(eta_)

        detTerm = -np.sum(np.log(L.diagonal()))
        diff = eta - mu

        beta_temp_kv = beta_doc_kv * np.exp(eta_)[:, None]
        bound = (
            np.log(
                theta[
                    None:,
                ]
                @ beta_temp_kv
            )
            @ word_count
            + detTerm
            - 0.5 * diff.T @ self.siginv @ diff
            - self.sigmaentropy
        )
        return bound

    def update_z(self, eta, beta_doc_kv, word_count):
        """Compute the update for the variational latent parameter z

        Args:
            eta (np.array): 1D-array representing prior to the document-topic distribution
            beta_doc_kv (np.array): 2D-array (K by V) containing the topic-word distribution for a specific document

        Returns:
            phi: update for the variational latent parameter z
        """
        eta_ = np.insert(eta, self.K - 1, 0)
        a = np.multiply(beta_doc_kv.T, np.exp(eta_)).T  # KxV
        b = np.multiply(a, (np.sqrt(word_count) / np.sum(a, 0)))  # KxV
        self.phi = np.multiply(b, np.sqrt(word_count).T)  # KxV
        assert np.all(self.phi >= 0), "Some values of phi are zero or nan."
        # return phi

    def save_model(self, output_dir):

        os.makedirs(output_dir, exist_ok=True)

        beta_output_path = os.path.join(output_dir, "beta_hat")
        np.save(beta_output_path, self.beta)

        theta_output_path = os.path.join(output_dir, "theta_hat")
        np.save(theta_output_path, self.theta)

        sigma_output_path = os.path.join(output_dir, "sigma_hat")
        np.save(sigma_output_path, self.sigma)

        eta_output_path = os.path.join(output_dir, "eta_hat")
        np.save(eta_output_path, self.eta)

        mu_output_path = os.path.join(output_dir, "mu_hat")
        np.save(mu_output_path, self.mu)

        X_output_path = os.path.join(output_dir, "X")
        np.save(X_output_path, self.X)

        if self.model == "STM":
            gamma_output_path = os.path.join(output_dir, "gamma_hat")
            np.save(gamma_output_path, self.gamma)

        lower_bound_path = os.path.join(output_dir, "lower_bound.pickle")

        with open(lower_bound_path, "wb") as f:
            pickle.dump(self.last_bounds, f)

    def label_topics(self, n, topics, frexweight=0.5, print_labels=False):
        """
        Label topics

        Generate a set of words describing each topic from a fitted STM object.

        Highest Prob: are the words within each topic with the highest probability
        (inferred directly from topic-word distribution parameter beta)
        FREX: weights exclusivity and frequency scores to get more meaningful topic labels.
        (Bischof and Airoldi 2012 for more details.)

        @param model STM model object.
        @param topics number of topics to include.  Default
        is all topics.
        @param n The desired number of words (per type) used to label each topic.
        Must be 1 or greater.

        TODO: @return labelTopics object (list) \item{prob }{matrix of highest
        probability words}
        """
        assert n >= 1, "n must be 1 or greater"

        if topics:
            K = topics
        else:
            K = self.K

        vocab = self.dictionary
        wordcounts = self.wcounts

        frex = self.frex(w=frexweight)

        # Sort by word probabilities on each row of beta
        # Returns words with highest probability per topic
        problabels = np.argsort(-1 * self.beta)[:n]
        frexlabels = np.argsort(-1 * frex)[:n]

        out_prob = []
        out_frex = []

        for k in K:
            probwords = [itemgetter(i)(vocab) for i in problabels[k, :n]]
            frexwords = [itemgetter(i)(vocab) for i in frexlabels[k, :n]]
            if print_labels:
                print(f"Topic {k}:\n \t Highest Prob: {probwords}")
                print(f"Topic {k}:\n \t FREX: {frexwords}")
            out_prob.append(probwords)
            out_frex.append(frexwords)

        return out_prob, out_frex

    def frex(self, w=0.5):
        """Calculate FREX (FRequency and EXclusivity) words
        A primarily internal function for calculating FREX words.
        Exclusivity is calculated by column-normalizing the beta matrix (thus representing the conditional probability of seeing
        the topic given the word).  Then the empirical CDF of the word is computed within the topic.  Thus words with
        high values are those where most of the mass for that word is assigned to the given topic.

        @param logbeta a K by V matrix containing the log probabilities of seeing word v conditional on topic k
        @param w a value between 0 and 1 indicating the proportion of the weight assigned to frequency

        """
        beta = np.log(self.beta)
        log_exclusivity = beta - sp.special.logsumexp(beta, axis=0)
        exclusivity_ecdf = np.apply_along_axis(self.ecdf, 1, log_exclusivity)
        freq_ecdf = np.apply_along_axis(self.ecdf, 1, beta)
        out = 1.0 / (w / exclusivity_ecdf + (1 - w) / freq_ecdf)
        return out

    def find_thoughts(self, topics, threshold=0, n=3):
        """
        Return the most prominent documents for a certain topic in order to identify representative
        documents. Topic representing documents might be conclusive underlying structure in the text
        collection.
        Following Roberts et al. (2016b):
        Theta captures the modal estimate of the proportion of word
        tokens assigned to the topic under the model.

        @param: threshold (np.float) minimal theta value of the documents topic proportion
            to be taken into account for the return statement.
        @param: topics to get the representative documents for
        @return: the top n document indices ranked by the MAP estimate of the topic's theta value

        Example: Return the 10 most representative documents for the third topic:
        > data.iloc[model.find_thoughts(topics=[3], n=10)]

        """
        assert n > 1, "Must request at least one returned document"
        if n > self.N:
            n = self.N

        for i in range(len(topics)):
            k = topics[i]
            # grab the values and the rank
            index = np.argsort(-1 * self.theta[:, k])[1:n]
            val = -np.sort(-1 * self.theta[:, k])[1:n]
            # subset to those values which meet the threshold
            index = index[np.where(val >= threshold)]
            # grab the document(s) corresponding to topic k
            return index

    def ecdf(self, arr):
        """Calculate the ECDF values for all elements in a 1D array."""
        return sp.stats.rankdata(arr, method="max") / arr.size
