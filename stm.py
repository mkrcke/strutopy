# %%

import json
import logging
import math
from pyexpat import model

# import matplotlib.pyplot as plt
import time

import numpy as np
import numpy.random as random
from pandas import Series
import scipy
import sklearn.linear_model
from scipy import optimize
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from operator import itemgetter

# custom packages
from generate_docs import CorpusCreation
from spectral_initialisation import spectral_init, create_dtm

logger = logging.getLogger(__name__)

class STM:
    def __init__(self, settings, documents, dictionary, dtype=np.float32, init='spectral', model='STM', mode='ols'):
        """
        @param: settings (c.f. large dictionary TODO: create settings file)
        @param: documents BoW-formatted documents in list of list of arrays with index-count tuples for each word
                example: `[[(1,3),(3,2)],[(1,1),(4,2)]]` -> [list of (int, int)]
        @param: dictionary contains word-indices of the corpus
        @param: dtype (default=np.float32) used for value checking along the process
        @param: init (default='spectral') init method to be used to initialise the word-topic distribution beta.  
                One might choose between 'random' and 'spectral', however the spectral initialisation is recommended
        @param: model (default='STM') to update variational mean parameter for the topical prevalence. Choose between 
                'STM' and 'CTM'. Note, however, that 'CTM' updates ignore the topical prevalence model.
        @param: mode (default='ols') to estimate the prevalence coefficients (gamma). Otherwise choose between l1 & l2 norm.

        @return:initialised values for the algorithm specifications parameters
                    - covar: topical prevalence covariates
                    # - enet: elastic-net configuration for the regularized update of the variational distribution over topics
                    - interactions: (bool) whether interactions between topics and covariates should be modelled (True) or not (False)
                    - betaindex: index for the topical prevalence covariate level (equals covar at the moment)
                    - last_bound: list to store approximated bound for each EM-iteration
                initialised values for the the global
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

        self.dtype = np.finfo(dtype).dtype

        # Specify Corpus & Settings
        # TODO: Unit test for corpus structure
        self.settings = settings
        self.documents = documents
        self.dictionary = dictionary
        self.init = init
        self.model = model
        self.mode = mode
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
        self.N = len(self.documents)
        self.covar = settings["covariates"]["X"]
        self.betaindex = settings["covariates"]["betaindex"]
        self.interactions = settings["kappa"]["interactions"]
        self.LDAbeta = settings["kappa"]["LDAbeta"]

        # convergence settings
        self.last_bounds = []
        self.max_em_its = settings["convergence"]["max.em.its"]

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
        if not self.LDAbeta: 
            self.init_kappa()
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
        if self.init=='spectral':
            self.beta = spectral_init(self.documents, self.K, self.V, maxV=5000)
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
            self.A,
            self.K,
            self.V,
        ), "Invalid beta shape. Got shape %s, but expected (%s, %s)" % (
            str(self.beta.shape),
            str(self.K),
            str(self.V),
        )

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
                    "Cholesky Decomposition failed, because Sigma is not positive definite."
                )
                self.sigmaentropy = (
                    0.5 * np.linalg.slogdet(self.sigma)[1]
                )  # part 2 of ELBO
                self.siginv = scipy.linalg.cholesky(self.sigma)  # part 3 of ELBO

        # initialize sufficient statistics
        calculated_bounds = []
        sigma_ss = np.empty_like(self.sigma)
        # sigma_ss = np.zeros(
        #     ((self.K - 1), (self.K - 1))
        # )  # update after each document optimization
        beta_ss = np.empty_like(self.beta)
        # if A != 1:
        #     beta_ss = np.zeros(self.K * self.V).reshape(
        #     self.K, self.V
        # )  # update after each document optimization
        # else:    
        #     beta_ss = np.zeros(self.K * self.V).reshape(
        #         self.K, self.V
        #     )  # update after each document optimization
        t1 = time.process_time()
        for i in range(self.N):

            # set document specs
            doc_array = np.array(self.documents[i])
            idx_1v = doc_array[
                :, 0
            ]  # This counts the first dimension of the numpy array, was "idx_1v"
            
            aspect = self.betaindex[i]

            beta_doc_kv = self.get_beta(idx_1v, aspect=aspect)
            word_count_1v = doc_array[:, 1]

            assert np.all(
                beta_doc_kv >= 0
            ), "Some entries of beta are negative.  Are you sure you didn't pass the logged version of beta?"

            # optimize variational posterior
            # does not matter if we use optimize.minimize(method='BFGS') or optimize fmin_bfgs()
            res = self.optimize_eta(
                eta=self.eta[i], mu=self.mu[i], word_count=word_count_1v, beta_doc=beta_doc_kv
            )

            # print(f"document {i}:", res.message)
            self.eta[i] = res.x
            self.theta[i] = np.exp(np.insert(res.x,self.K-1,0))/np.sum(np.exp(np.insert(res.x,self.K-1,0)))
            # Compute Hessian, Phi and Lower Bound
            # 1) check if inverse is a legitimate cov matrix
            # 2) if not, adjust matrix to be positive definite
            hess_i = self.hessian(
                eta=self.eta[i],
                word_count=word_count_1v,
                beta_doc_kv=beta_doc_kv
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
            phi = self.update_z(
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
            beta_doc_kv = self.beta[aspect][:, np.array(np.int0(words))]
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

    def update_mu(self, intercept=False):
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
            #assert self.A < 2, 'Uses column means for the mean, since no covariates are specified.'
            # just use the mean for all documents 
            self.mu = np.repeat(np.mean(self.eta, axis=0)[None,:], self.N, axis=0)
         
        # mode = L1 simplest method requires only glmnet (https://cran.r-project.org/web/packages/glmnet/index.html)
        elif self.model == "STM":
            #prepare covariate matrix for modeling 
            try:
                self.covar = self.covar.astype('category')
            except:
                pass
            
            prev_cov = np.array(self.covar)[:,None] #prepares 1D array for one-hot encoding (OHE) by making it 2D
            
            # remove empty dimension
            if len(prev_cov.shape)>2: 
                prev_cov = np.squeeze(prev_cov, axis=1)
            
            if not np.array_equal(prev_cov, prev_cov.astype(bool)):
                enc = OneHotEncoder(handle_unknown='ignore') #create OHE
                prev_cov = enc.fit_transform(prev_cov).toarray() #fit OHE
            
            if self.mode not in ['lasso','ridge', 'ols']: 
                print("Need to specify the estimation mode of prevalence covariate coefficients. Uses default 'ols'.")
 
            if self.mode == 'lasso':
                linear_model = sklearn.linear_model.Lasso(alpha=1, fit_intercept=intercept)
                fitted_model = linear_model.fit(prev_cov,self.eta)
            
            elif self.mode == 'ridge':
                linear_model = sklearn.linear_model.Ridge(alpha=.1, fit_intercept=intercept)
                fitted_model = linear_model.fit(prev_cov,self.eta)
            
            else:
                linear_model = sklearn.linear_model.LinearRegression(fit_intercept=intercept)
                fitted_model = linear_model.fit(prev_cov, self.eta)
            
            # adjust design matrix if intercept is estimated
            if intercept: 
                self.gamma = np.column_stack((fitted_model.intercept_, fitted_model.coef_))
                design_matrix = np.c_[ np.ones(prev_cov.shape[0]), prev_cov]

            self.gamma = fitted_model.coef_
            design_matrix = prev_cov
            
            self.mu = design_matrix@self.gamma.T
        
        else: 
            raise ValueError('Updating the topical prevalence parameter requires a mode. Choose from "CTM", "Pooled" or "L1" (default).')
    

    def update_sigma(self, nu, sigprior):
        """
        Updates the variance covariance matrix for the logistic normal distribution of topical prevalence

        Args:
            nu (_type_): variance-covariance for the variational document-topic distribution
            sigprior (_type_): prior for the var-cov. matrix for the log-normal
        """
        # find the covariance
        covariance = (self.eta - self.mu).T @ (self.eta - self.mu)
        sigma = (covariance + nu)/self.N
        self.sigma = np.diag(np.diag(sigma)) * sigprior + (1 - sigprior) * sigma

    def update_beta(self, beta_ss):
        """
        Updates the topic-word distribution

        Args:
            kappa (_type_, optional): topical content covariate. Defaults to None.
        """
        # computes the update for beta based on the SAGE model
        # for now: just computes row normalized beta values as a point estimate 
        if self.LDAbeta:
            self.beta = beta_ss / np.sum(beta_ss, axis=1)[:, None]
        else:
            self.mnreg(beta_ss=beta_ss)
    
    def mnreg(self, beta_ss): 
        """estimation of distributed poisson regression for the update of the kappa parameters

        @param: beta_ss (np.ndarray) estimated word-topic distribution of the current EM-iteration with dimension K x V
        """
        
        contrast = False
        interact = True
        fixed_intercept = True
        alpha = 250 #corresponds to `lambda` in glmnet 
        maxit=1e4
        tol=1e-5


        counts = csr_matrix(np.concatenate((beta_ss[0],beta_ss[1]),axis=0)) # dimension (A*K) x V # TODO: enable dynamic creation of 'counts'

        # Three cases
        if self.A == 1: # Topic Model
            covar = np.diag(np.ones(self.K))
        if self.A != 1: # Topic-Aspect Models
            # if not contrast: 
            #Topics
            veci = np.arange(0,counts.shape[0])
            vecj = np.tile(np.arange(0, self.K), self.A)
            #aspects
            veci = np.concatenate((veci, np.arange(0, (counts.shape[0]))))
            vecj = np.concatenate((vecj, np.repeat(np.arange(self.K+1,self.K+self.A+1), self.K)))
            if interact: 
                veci = np.concatenate((veci, np.arange(0,counts.shape[0])))
                vecj = np.concatenate((vecj, np.arange(self.K+self.A+1, self.K+self.A+counts.shape[0]+1))) #TODO: remove +1 at the end, make shapes fit anyway
            vecv = np.ones(len(veci))
            covar = csr_matrix((vecv, (veci,vecj))) 

        if fixed_intercept: 
            m = self.wcounts
            m = np.log(m) - np.log(np.sum(m))
        else: 
            m = 0 
        
        mult_nobs = counts.sum(axis=1)  
        offset = np.log(mult_nobs)
        #counts = np.split(counts, counts.shape[1], axis=1)


        ############################
        ### Distributed Poissons ###
        ############################
        out = []
        #now iterate over the vocabulary
        for i in range(counts.shape[1]):
            
            if np.all(m==0): 
                offset2 = offset
                fit_intercept=True
            else: 
                fit_intercept=False
            offset2 = m[i] + offset
            mod = None
            #while mod is None: 
            # alpha = alpha * np.floor(0.2*alpha)
            clf = sklearn.linear_model.PoissonRegressor(fit_intercept=fit_intercept, max_iter=np.int0(maxit), tol=tol, alpha=np.int0(alpha))
            mod = clf.fit(covar, counts[:,[1]].A.flatten())
                #if it didn't converge, increase nlambda paths by 20% 
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

        linpred = covar@coef
        linpred = m + linpred
        explinpred = np.exp(linpred)
        beta =  explinpred/np.sum(explinpred, axis=1)[:,np.newaxis]
        
        # retain former structure for beta
        self.beta = np.split(beta, 2, axis=0)

    def expectation_maximization(self, saving, prefix):
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
                    self.save_model(prefix)
                break
            if self.max_its_reached(_iteration):
                self.time_processed = time.process_time() - t1
                print(
                    f"maximum number of iterations ({self.max_em_its}) reached after {self.time_processed} seconds"
                )
                if saving == True: 
                    print("saving model...")
                    self.save_model(prefix)
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

    def optimize_eta(self, eta, mu,  word_count, beta_doc):
        """Optimizes the variational parameter eta given the likelihood and the gradient function"""

        def f(eta, word_count, mu, beta_doc):
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
            return np.float64((0.5 * (eta[:-1] - mu).T @ self.siginv @ (eta[:-1] - mu)) - (np.dot(
                word_count, eta.max() + np.log(np.exp(eta - eta.max()) @ beta_doc))
            - Ndoc * scipy.special.logsumexp(eta)))

        def df(eta, word_count, mu, beta_doc):
            """gradient for the objective of the variational update q(etas)"""
            eta = np.insert(eta, self.K - 1, 0)
            # formula
            return np.array(np.float64(self.siginv @ (eta[:-1] - mu)-(beta_doc @ (word_count / np.sum(beta_doc.T, axis=1))
            - (np.sum(word_count) / np.sum(np.exp(eta)))*np.exp(eta))[:-1]))

        return optimize.minimize(
            f, x0=eta, args=(word_count,mu,beta_doc), jac=df, method="BFGS"
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

    def save_model(self, prefix):
        model = {
            "bound": self.last_bounds,
            "mu": self.mu,
            "sigma": self.sigma,
            "beta": self.beta,
            "time_processed": self.time_processed,
            "lambda":self.eta,
            "theta":self.theta, 
            #"settings": self.settings,
            #"documents": self.documents,
        }
        if prefix:
            json.dump(model, open(f"fitted_models/{prefix}_model.json", "w"), cls=NumpyEncoder)
        else: 
            json.dump(model, open("fitted_models/model.json", "w"), cls=NumpyEncoder)
    
    def label_topics(self, n, topics):
        """
        Label topics
        
        Generate a set of words describing each topic from a fitted STM object.
        
        Highest Prob: are the words within each topic with the highest probability
        (inferred directly from topic-word distribution parameter beta)
        
        @param model STM model object.
        @param topics number of topics to include.  Default
        is all topics.
        @param n The desired number of words (per type) used to label each topic.
        Must be 1 or greater.

        TODO: @return labelTopics object (list) \item{prob }{matrix of highest
        probability words}
        """
        assert n>=1, 'n must be 1 or greater'

        if topics:
            K = topics
        else: 
            K = self.K
        
        vocab = self.dictionary

        # wordcounts = model.settings["dim"]["wcounts"]["x"] #TODO: implement word counts
        
        # Sort by word probabilities on each row of beta
        # Returns words with highest probability per topic
        problabels = np.argsort(-1*self.beta)[:n]

        out = []
        for k in range(K):
            probwords = [itemgetter(i)(vocab) for i in problabels[k,:n]]
            print(f"Topic {k}:\n \t Highest Prob: {probwords}")
            out.append(probwords)
        
        return


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


# %%
