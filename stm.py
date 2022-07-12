# %%

import json
import logging
import math

# import matplotlib.pyplot as plt
import time

import numpy as np
import numpy.random as random
from pandas import Series
import scipy
import sklearn.linear_model
from scipy import optimize
from sklearn.preprocessing import OneHotEncoder

# from stm import STM
from generate_docs import CorpusCreation
#from spectral_initialisation import spectral_init

# custom packages


logger = logging.getLogger(__name__)


# %%


class STM:
    def __init__(self, settings, documents, dictionary, dtype=np.float32):
        """
        @param: settings (c.f. large dictionary TODO: create settings file)
        @param: documents BoW-formatted documents in list of list of arrays with index-count tuples for each word
                example: `[[(1,3),(3,2)],[(1,1),(4,2)]]` -> [list of (int, int)]
        @param: dictionary contains word-indices of the corpus
        @param: (default=np.float32) dtype used for value checking along the process

        @return:initialised values for the algorithm specifications parameters
                    - covar: topical prevalence covariates
                    - enet: elastic-net configuration for the regularized update of the variational distribution over topics
                    - interactions: (bool) whether interactions between topics and covariates should be modelled (True) or not (False)
                    - betaindex: index for the topical prevalence covariate level (equals covar at the moment)
                    - last_bound: list to store approximated bound for each EM-iteration
                initialised values for the the global
                    - V: number of tokens
                    - K: number of topics 
                    - N: number of documents 
                    - A: topical prevalence covariate levels
                    - mu: prior on topical prevalence
                    - sigma: prior covariance on topical prevalence
                    - beta: prior on word-topic distribution
                    - eta: prior on document-topic distribution
                    - kappa: prior on topical content covariates
                    - lamda: prior on the variational distribution of the topical content model
        """

        self.dtype = np.finfo(dtype).dtype

        # Specify Corpus & Settings
        # TODO: Unit test for corpus structure
        self.settings = settings
        self.documents = documents
        self.dictionary = dictionary
        # TODO: create a document term matrix (D x V) that is required for the spectral initialisation
        #self.doc_term_matrix = 

        self.init = settings['init']['mode']
        
        # test and store user-supplied parameters
        if len(documents) is None:
            raise ValueError("documents must be specified to establish input space")
        if self.settings["dim"]["K"] == 0:
            raise ValueError("Number of topics must be specified")
        if self.settings["dim"]["A"] == 1:
            logger.warning("no dimension for the topical content provided")

        self.V = self.settings["dim"]["V"]  # Number of words
        self.K = self.settings["dim"]["K"]  # Number of topics
        self.A = self.settings["dim"]["A"]  # TODO: when data changes ...
        self.covar = settings["covariates"]["X"]
        self.enet = settings["tau"]["enet"]
        self.N = len(self.documents)
        self.interactions = settings["kappa"]["interactions"]
        self.betaindex = settings["covariates"]["betaindex"]

        self.last_bounds = []

        self.init_global_params()  # Think about naming. Are these global params?

    # _____________________________________________________________________
    def init_global_params(self):
        """initialises global parameters beta, mu, eta, sigma and kappa"""
        # Set global params
        self.init_beta()
        self.init_mu()
        self.init_eta()
        self.init_sigma()
        self.init_kappa()
        self.init_lamda()

    def init_beta(self):
        """
        beta : {float, numpy.ndarray of float, list of float, str}, optional
            A-priori belief on topic-word distribution, this can be:
                * scalar for a symmetric prior over topic-word distribution,
                * 1D array of length equal to num_words to denote an asymmetric user defined prior for each word,
                * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination.
        """
        if self.init=='spectral':
            self.beta = spectral_init(self.doc_term_matrix, self.K, maxV=10000)
        elif self.init=='random':
            beta_init = random.gamma(0.1, 1, self.V * self.K).reshape(self.K, self.V)
            beta_init_normalized = beta_init / np.sum(beta_init, axis=1)[:, None]
            if self.interactions:  # TODO: replace ifelse condition by logic
                self.beta = np.repeat(beta_init_normalized[None, :], self.A, axis=0)
                # test if probabilities sum to 1
                [
                    np.testing.assert_almost_equal(sum_over_words, 1)
                    for i in range(self.A)
                    for sum_over_words in np.sum(self.beta[i], axis=1)
                ]
            else:
                self.beta = beta_init_normalized
                [
                    np.testing.assert_almost_equal(sum_over_words, 1)
                    for i in range(self.A)
                    for sum_over_words in np.sum(self.beta, axis=1)
                ]
        assert self.beta.shape == (
            self.K,
            self.V,
        ), "Invalid beta shape. Got shape %s, but expected %s" % (
            str(self.beta.shape),
            str(self.K, self.V),
        )

    # TODO: Check for the shape of mu if this is correct
    def init_mu(self):
        self.mu = np.zeros((self.K - 1,))

    def init_sigma(self):
        self.sigma = np.zeros(((self.K - 1), (self.K - 1)))
        np.fill_diagonal(self.sigma, 20)
    

    def init_eta(self):
        """
        dimension: N by K-1
        """
        self.eta = np.zeros((self.N, self.K - 1))

    def init_lamda(self):
        """document level parameter to store the mean variational parameters
        based on the numerical optimization and the log additive transformation.
        dimension: N by K
        """
        self.lamda = np.zeros((self.N, self.K))

    def init_gamma(self): 
        """The prior specification for the topic prevalence parameters is a zero mean Gaussian distribution with shared variance parameter,
        gamma_p,k ~ N(0,sigma_k^2)
        sigma_k^2 ~ InverseGamma(a,b), with a & b fixed
        """
        self.gamma = np.zeros(self.A, self.K) 

    def init_kappa(self):
        """
        Initializing Topical Content Model Parameters
        """
        # read in documents and vocab
        flat_documents = [item for sublist in self.documents for item in sublist]
        m = []

        total_sum = sum(n for _, n in flat_documents)

        for elem in flat_documents:
            m.append(elem[1] / total_sum)
        m = np.log(m) - np.log(np.mean(m))  # logit of m
        # Defining parameters
        aspectmod = (
            self.A > 1
        )  # if there is more than one level for the topical content
        if aspectmod:
            interact = self.interactions  # allow for the choice to interact
        else:
            interact = False
        # Create the parameters object
        parLength = self.K + self.A * aspectmod + (self.K * self.A) * interact
        # create covariates. one element per item in parameter list.
        # generation by type because its conceptually simpler
        if not aspectmod & interact:
            covar = {
                "k": np.arange(self.K),
                "a": np.repeat(np.nan, parLength),  # why parLength?
                "type": np.repeat(1, self.K),
            }

        if aspectmod & interact == False:
            covar = {
                "k": np.append(np.arange(self.K), np.repeat(np.nan, self.A)),
                "a": np.append(np.repeat(np.nan, self.K), np.arange(self.A)),
                "type": np.append(np.repeat(1, self.K), np.repeat(2, self.A)),
            }
        if interact:
            covar = {
                "k": np.append(
                    np.arange(self.K),
                    np.append(
                        np.repeat(np.nan, self.A), np.repeat(np.arange(self.K), self.A)
                    ),
                ),
                "a": np.append(
                    np.repeat(np.nan, self.K),
                    np.append(np.arange(self.A), np.repeat(np.arange(self.A), self.K)),
                ),
                "type": np.append(
                    np.repeat(1, self.K),
                    np.append(np.repeat(2, self.A), np.repeat(3, self.K * self.A)),
                ),
            }

        self.kappa_initialized = {
            "m": m,
            "params": np.tile(np.repeat(0, self.V), (parLength, 1)),
            "covar": covar
            #'kappasum':, why rolling sum?
        }

    # _____________________________________________________________________
    def E_step(self):
        """
        updates the sufficient statistics for each e-step iteration:
            sigma_ss: np.array of shape ((k-1), (k-1))
            beta_ss: np.array of shape (k, v) but might need A
            bound: might be implemented as list len(#iterations)
        """
        iter = 0
        # 2) Precalculate common components
        while True:
            try:
                sigobj = np.linalg.cholesky(
                    self.sigma
                )
                self.sigmaentropy = np.sum(np.log(np.diag(sigobj)))
                self.siginv = np.linalg.inv(sigobj).T * np.linalg.inv(sigobj)
                break
            except:
                print(
                    "Cholesky Decomposition failed, because Sigma is not positive definite!"
                )
                self.sigmaentropy = (
                    0.5 * np.linalg.slogdet(self.sigma)[1]
                )  # part 2 of ELBO
                self.siginv = scipy.linalg.cholesky(self.sigma)  # part 3 of ELBO

        # initialize sufficient statistics
        calculated_bounds = []
        sigma_ss = np.zeros(
            ((self.K - 1), (self.K - 1))
        )  # update after each document optimization
        beta_ss = np.zeros(self.K * self.V).reshape(
            self.K, self.V
        )  # update after each document optimization
        t1 = time.process_time()
        for i in range(self.N):

            # set document specs
            doc_array = np.array(self.documents[i])
            idx_1v = doc_array[
                :, 0
            ]  # This counts the first dimension of the numpy array, was "idx_1v"
            aspect = self.betaindex[i]
            beta_doc_kv = self.get_beta(idx_1v, aspect)
            word_count_1v = doc_array[:, 1]

            assert np.all(
                beta_doc_kv >= 0
            ), "Some entries of beta are negative.  Are you sure you didn't pass the logged version of beta?"

            # optimize variational posterior
            # does not matter if we use optimize.minimize(method='BFGS') or optimize fmin_bfgs()
            res = self.optimize_eta(
                eta=self.eta[i], word_count=word_count_1v, beta_doc=beta_doc_kv
            )

            # print(f"document {i}:", res.message)
            self.eta[i] = res.x
            self.lamda[i] = np.exp(np.insert(res.x, self.K-1, 0)) / np.sum(np.exp(np.insert(res.x, self.K-1, 0)))
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
                mu=self.mu,
                word_count=word_count_1v,
                beta_doc_kv=beta_doc_kv,
                eta=self.eta[i],
            )

            # Delta NU
            nu = self.optimize_nu(L_i)

            # Delta Phi
            phi = self.update_z(
                eta=self.eta[i], beta_doc_kv=beta_doc_kv, word_count=word_count_1v
            )

            # print(bound_i)
            calculated_bounds.append(bound_i)

            # update sufficient statistics
            sigma_ss += nu
            # TODO: combine into one
            if self.interactions:
                beta_ss[aspect][:, np.array(np.int0(idx_1v))] += phi
            else:
                beta_ss[:, np.array(np.int0(idx_1v))] += phi

        self.bound = np.sum(calculated_bounds)
        self.last_bounds.append(self.bound)

        
        print(f'Lower Bound: {self.bound}')
        print(
            "Completed E-Step in ({} seconds). \n".format(
                math.floor((time.process_time() - t1))
            )
        )
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
            beta_doc_kv = self.beta[aspect][:, np.array(words)]
        else:
            beta_doc_kv = self.beta[:, np.array(np.int0(words))]
        return beta_doc_kv

    # _____________________________________________________________________
    def M_step(self, beta_ss, sigma_ss):
        # Run M-Step

        t1 = time.process_time()

        self.update_mu()

        self.update_sigma(nu=sigma_ss, sigprior=self.settings["sigma"]["prior"])

        self.update_beta(beta_ss)

        print(
            "Completed M-Step ({} seconds). \n".format(
                math.floor((time.process_time() - t1))
            )
        )
        print('___________________________________________________')

    def update_mu(self, mode = "CTM"):
        """
        updates the mean parameter for the [document specific] logistic normal distribution
        """
        if mode == "CTM":
            #assert self.A < 2, 'Uses column means for the mean, since no covariates are specified.'
            self.mu = np.mean(self.eta, axis=0)
         
        # mode = L1 simplest method requires only glmnet (https://cran.r-project.org/web/packages/glmnet/index.html)
        elif mode == "L1":
            #prepare covariate matrix for modeling 
            try:
                covar = self.covar.astype('category')
            except:
                pass
            covar2D = np.array(self.covar)[:,None] #prepares 1D array for one-hot encoding (OHE) by making it 2D
            enc = OneHotEncoder(handle_unknown='ignore') #create OHE
            covarOHE = enc.fit_transform(covar2D).toarray() #fit OHE
            linear_model = sklearn.linear_model.Lasso(alpha=enet)
            fitted_model = linear_model.fit(covarOHE,self.lambd)
            self.gamma = np.insert(fitted_model.coef_, 0, fitted_model.intercept_).reshape(self.K-1,3)
            design_matrix = np.c_[ np.ones(covarOHE.shape[0]), covarOHE]
            self.mu = design_matrix@self.gamma.T
        else: 
            raise ValueError('Updating the topical prevalence parameter requires a mode. Choose from "CTM", "Pooled" or "L1" (default).')
    

    def update_sigma(self, nu, sigprior):
        """
        Updates the variance covariance matrix for the logistic normal distribution

        Args:
            nu (_type_): variance-covariance for the variational document-topic distribution
            sigprior (_type_): prior for the var-cov. matrix for the log-normal
        """
        # find the covariance
        covariance = (self.eta - self.mu).T @ (self.eta - self.mu)
        sigma = (covariance + nu)/self.N
        self.sigma = np.diag(np.diag(sigma)) * sigprior + (1 - sigprior) * sigma

    def update_beta(self, beta_ss, kappa=None):
        """
        Updates the topic-word distribution

        Args:
            kappa (_type_, optional): topical content covariate. Defaults to None.
        """
        # computes the update for beta based on the SAGE model
        # for now: just computes row normalized beta values
        if kappa is None:
            self.beta = beta_ss / np.sum(beta_ss, axis=1)[:, None]
        else:
            print(f"implementation for {kappa} is missing")

    def expectation_maximization(self, saving):
        t1 = time.process_time()
        for _iteration in range(100):
            print(f'________________Iteration:{_iteration}_____________________')
            beta_ss, sigma_ss = self.E_step()
            self.M_step(beta_ss, sigma_ss)
            if self.EM_is_converged(_iteration):
                self.time_processed = time.process_time() - t1
                print(
                    f"model converged in iteration {_iteration} after {self.time_processed}s"
                )
                if saving == True: 
                    print("saving model...")
                    self.save_model()
                break
            if self.max_its_reached(_iteration):
                self.time_processed = time.process_time() - t1
                print(
                    f"maximum number of iterations ({max_em_its}) reached after {self.time_processed} seconds"
                )
                if saving == True: 
                    print("saving model...")
                    self.save_model()
                break 

    # _____________________________________________________________________
    def EM_is_converged(self, _iteration, convergence=None):

        if _iteration < 2:
            return False

        new = self.bound
        old = self.last_bounds[-2]
        emtol = self.settings["convergence"]["em.converge.thresh"]

        convergence_check = np.abs((new - old) / np.abs(old))
        print(f"relative change: {convergence_check}")
        if convergence_check < emtol:
            return True
        else:
            return False

    def max_its_reached(self, _iteration):
        if _iteration == max_em_its - 1:
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

    def optimize_eta(self, eta, word_count, beta_doc):
        """Optimizes the variational parameter eta given the likelihood and the gradient function"""

        def f(eta, word_count, beta_doc):
            """objective for the variational update q(eta)

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
            # formula
            # from cpp implementation:
            # log(expeta * betas) * doc_cts - ndoc * log(sum(expeta))
            return np.float64((0.5 * (eta[:-1] - self.mu).T @ self.siginv @ (eta[:-1] - self.mu)) - (np.dot(
                word_count, eta.max() + np.log(np.exp(eta - eta.max()) @ beta_doc))
            - Ndoc * scipy.special.logsumexp(eta)))

        def df(eta, word_count, beta_doc):
            """gradient for the objective of the variational update q(etas)"""
            eta = np.insert(eta, self.K - 1, 0)
            # formula
            return np.array(np.float64(self.siginv @ (eta[:-1] - self.mu)-(beta_doc @ (word_count / np.sum(beta_doc.T, axis=1))
            - (np.sum(word_count) / np.sum(np.exp(eta)))*np.exp(eta))[:-1]))

        return optimize.minimize(
            f, x0=eta, args=(word_count, beta_doc), jac=df, method="BFGS"
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
        eta_ = np.insert(eta, self.K - 1, 0)
        theta = self.stable_softmax(eta_)
        
        a = np.transpose(np.multiply(np.transpose(beta_doc_kv), np.exp(eta_)))  # KxV
        b = np.multiply(a, np.transpose(np.sqrt(word_count))) / np.sum(a, 0)  # KxV
        c = np.multiply(b, np.transpose(np.sqrt(word_count)))  # KxV

        hess = b @ b.T - np.sum(word_count) * np.multiply(
            theta[:,None], theta[None,:]
        )
        assert self.check_symmetric(hess), 'Hessian is not symmetric'
        # broadcasting, works fine
        # difference to the c++ implementation comes from unspecified evaluation order: (+) instead of (-)
        np.fill_diagonal(
            hess, np.diag(hess) - np.sum(c, axis=1) + np.sum(word_count)*theta
        )

        d = hess[:-1, :-1]
        f = d + self.siginv   
        
        if not np.all(np.linalg.eigvals(f)>0):
            print('Hessian not positive definite. Introduce Diagonal Dominance...')
            f = self.make_pd(f)

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
                print("converts Hessian via diagonal-dominance")
            except:
                L = scipy.linalg.cholesky(
                    self.make_pd(hess) + 1e-5 * np.eye(hess.shape[0])
                )
                print("adds a small number to the hessian")
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
        nu = np.dot(nu,np.transpose(nu))
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
        # compute 1/2 the determinant from the cholesky decomposition
        detTerm = -np.sum(np.log(L.diagonal()))
        diff = eta - mu
        ############## generate the bound and make it a scalar ##################
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
        phi = np.multiply(b, np.sqrt(word_count).T)  # KxV
        return phi

    def save_model(self):
        model = {
            "bound": self.last_bounds,
            "mu": self.mu,
            "sigma": self.sigma,
            "beta": self.beta,
            #"settings": self.settings,
            "time_processed": self.time_processed,
            "lambda":self.lamda,
            "documents": self.documents,
        }
        json.dump(model, open("model.json", "w"), cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating): 
            return float(obj)
        elif isinstance(obj, np.floating): 
            return float(obj)
        elif isinstance(obj, Series):
            return obj.to_json()
        return json.JSONEncoder.default(self, obj)


# %% Init params for training _____________________

# Parameter Settings (required for simulation process)
num_topics = 10
A = 2
verbose = True
interactions = False  # settings.kappa

# Initialization and Convergence Settings
init_type = "Random"  # settings.init
ngroups = 1  # settings.ngroups
max_em_its = 30  # settings.convergence
emtol = 1e-5  # settings.convergence


# Here we are simulating 100 documents with 100 words each. We are sampling from a multinomial distribution with dimension V.
# Note however that we will discard all elements from the vector V that do not occur.
# This leads to a dimension of the vocabulary << V
np.random.seed(123)

Corpus = CorpusCreation(
    n_topics=num_topics,
    n_docs=100,
    n_words=100,
    V=250,
    treatment=False,
    alpha='symmetric',
)
Corpus.generate_documents()
betaindex = np.concatenate(
    [np.repeat(0, len(Corpus.documents) / 2), np.repeat(1, len(Corpus.documents) / 2)]
)



# %%
for doc in Corpus.documents:
    word_idx = np.array(doc)[0]
    word_count = np.array(doc)[1]