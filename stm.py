# %%

import json
import logging
import math
#import matplotlib.pyplot as plt
import time

import numpy as np
import numpy.random as random
import scipy
import sklearn.linear_model
from scipy import optimize
from sklearn.preprocessing import OneHotEncoder

#from stm import STM
from generate_docs import CorpusCreation

# custom packages




logger = logging.getLogger(__name__)



# %%

class STM:

    def __init__(self, settings, documents, dictionary, dtype=np.float32):
        """
            beta : {float, numpy.ndarray of float, list of float, str}, optional
            A-priori belief on topic-word distribution, this can be:
                * scalar for a symmetric prior over topic-word distribution,
                * 1D array of length equal to num_words to denote an asymmetric user defined prior for each word,
                * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination.
        """

        self.dtype = np.finfo(dtype).dtype

        # Specify Corpus & Settings
        # TODO: Unit test for corpus structure
        self.settings=settings
        self.documents = documents
        self.dictionary = dictionary

        # test and store user-supplied parameters
        if len(documents) is None: 
            raise ValueError(
                'documents must be specified to establish input space'
            )
        if self.settings['dim']['K'] == 0:
            raise ValueError("Number of topics must be specified")
        if self.settings['dim']['A'] == 1:
            logger.warning("no dimension for the topical content provided")

        self.V = self.settings['dim']['V'] # Number of words
        self.K = self.settings['dim']['K'] # Number of topics
        self.A = self.settings['dim']['A'] # TODO: when data changes ...
        self.N = len(self.documents)
        self.interactions = settings['kappa']['interactions']
        self.betaindex = settings['covariates']['betaindex']
        
        # initialize bound? 
        self.last_bounds = [0.00001]

        self.init_global_params() # Think about naming. Are these global params?

    # _____________________________________________________________________
    def init_global_params(self):
        """initialises global parameters beta, mu, eta, sigma and kappa
        """
        # Set global params
        self.init_beta()
        self.init_mu()
        self.init_eta()
        self.init_sigma()
        self.init_kappa()


    def init_beta(self):
        """ Beta has shape str(self.K, self.V))
        """

        beta_init = random.gamma(.1,1, self.V*self.K).reshape(self.K,self.V)
        beta_init_normalized = (beta_init / np.sum(beta_init, axis=1)[:,None])
        if self.interactions: #TODO: replace ifelse condition by logic
            self.beta = np.repeat(beta_init_normalized[None,:], self.A, axis=0)
            # test if probabilities sum to 1      
            [np.testing.assert_almost_equal(sum_over_words, 1) for i in range(self.A) for sum_over_words in np.sum(self.beta[i], axis=1)]
        else: 
            self.beta = beta_init_normalized
            [np.testing.assert_almost_equal(sum_over_words, 1) for i in range(self.A) for sum_over_words in np.sum(self.beta, axis=1)]
        assert self.beta.shape == (self.K,self.V), \
            "Invalid beta shape. Got shape %s, but expected %s" % (str(self.beta.shape), str(self.K, self.V))

    # TODO: Check for the shape of mu if this is correct
    def init_mu(self):
        self.mu = np.zeros((self.K-1, ))

    def init_sigma(self):
        self.sigma = np.zeros(((self.K-1),(self.K-1)))
        np.fill_diagonal(self.sigma, 20)

    def init_eta(self):
        """
        dimension: N by K-1
        """
        self.eta = np.zeros((self.N, self.K-1))

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
        m = np.log(m) - np.log(np.mean(m)) #logit of m
        #Defining parameters
        aspectmod = self.A > 1 # if there is more than one level for the topical content
        if(aspectmod):
            interact = self.interactions # allow for the choice to interact
        else:
            interact = False
        #Create the parameters object
        parLength = self.K + self.A * aspectmod + (self.K*self.A)*interact
        #create covariates. one element per item in parameter list.
        #generation by type because its conceptually simpler
        if not aspectmod & interact:
            covar = {'k': np.arange(self.K),
                'a': np.repeat(np.nan, parLength), #why parLength? 
                'type': np.repeat(1, self.K)}

        if(aspectmod & interact == False):
            covar = {'k': np.append(np.arange(self.K), np.repeat(np.nan, self.A)),
                    'a': np.append(np.repeat(np.nan, self.K), np.arange(self.A)), 
                    'type': np.append(np.repeat(1, self.K), np.repeat(2, self.A))}      
        if(interact):
            covar = {'k': np.append(np.arange(self.K), np.append(np.repeat(np.nan, self.A), np.repeat(np.arange(self.K), self.A))),
                    'a': np.append(np.repeat(np.nan, self.K), np.append(np.arange(self.A), np.repeat(np.arange(self.A), self.K))), 
                    'type': np.append(np.repeat(1, self.K), np.append(np.repeat(2, self.A),  np.repeat(3,self.K*self.A)))}

        self.kappa_initialized = {'m':m,
                        'params' : np.tile(np.repeat(0,self.V), (parLength, 1)),
                        'covar' : covar
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
        # 2) Precalculate common components
        while True:
            try: 
                sigobj = np.linalg.cholesky(self.sigma) #initialization of sigma not positive definite
                self.sigmaentropy = np.sum(np.log(np.diag(sigobj)))
                self.siginv = np.linalg.inv(sigobj).T*np.linalg.inv(sigobj)
                break
            except:
                print("Cholesky Decomposition failed, because Sigma is not positive definite!")
                self.sigmaentropy = .5*np.linalg.slogdet(self.sigma)[1] # part 2 of ELBO 
                self.siginv = np.linalg.solve(self.sigma)           # part 3 of ELBO

        calculated_bounds = []

        # V, A , K , N, mu, docs, lambda, eta, beta, betacov == y, 
    
        for i in range(self.N):

            eta_extend = np.insert(self.eta[i], self.K-1, 0)

            #set document specs
            doc = self.documents[i] # TODO: Make documents a numpy array

            doc_array = np.array(doc)

            idx_1v = doc_array[:, 0] # This counts the first dimension of the numpy array, was "idx_1v"
            aspect = self.betaindex[i]
            beta_doc_kv = self.get_beta(idx_1v, aspect)

            assert np.all(beta_doc_kv >= 0), \
                "Some entries of beta are negative.  Are you sure you didn't pass the logged version of beta?"

            # This does not make sense.
            # word_count_1v = np.array([y for x,y in doc]) #count of words in document
            word_count_1v = doc_array[:, 1]
            Ndoc = np.sum(word_count_1v)

            # initial values
            theta_1k = self.stable_softmax(eta_extend)

            phi_vk = self.softmax_weights(eta_extend, beta_doc_kv)
            
            # optimize variational posterior
            # does not matter if we use optimize.minimize(method='BFGS') or optimize fmin_bfgs()

            # x0 : ndarray, shape (n,)
            #     Initial guess. Array of real elements of size (n,),
            #     where ``n`` is the number of independent variables.
            # args : tuple, optional

            # We want to maximize f, but numpy only implements minimize, so we
            # minimize -f
            # word_count_1v, eta_long, beta_doc_kv, Ndoc, phi_vk, theta_1k, self.K-1
            res = self.optimize_eta(
                eta=self.eta[i],
                word_count=word_count_1v,
                eta_extend=eta_extend,
                beta_doc=beta_doc_kv,
                Ndoc=Ndoc,
                phi=phi_vk,
                theta=theta_1k
                )
            print(f"document {i}:", res.message)
            self.eta[i] = res.x
            
            # Compute Hessian, Phi and Lower Bound 

            # hess_inv_i = res.hess_inv # TODO: Make a self.inverted_hessian[i] # TODO: replace approximation with analytically derived Hessian

            # 1) check if inverse is a legitimate cov matrix
            # 2) if not, adjust matrix to be positive definite 
            
            hess_i = self.compute_hessian(eta = self.eta[i], word_count=word_count_1v, beta_doc_kv=beta_doc_kv)
            L_i = self.decompose_hessian(hess_i)

            # Delta NU
            nu = self.optimize_nu(L_i)

            # Delta Bound
            bound_i = self.lower_bound(
                L_i,
                mu=self.mu,
                word_count=word_count_1v,
                beta_doc_kv=beta_doc_kv,
                eta=self.eta[i]
            )
            # Delta Phi
            phi = self.update_z(
                eta = self.eta[i],
                beta_doc_kv=beta_doc_kv)
            
            #print(bound_i)

            calculated_bounds.append(bound_i)

            self.sigma += nu

            #TODO: combine into one    
            if self.interactions:
                self.beta[aspect][:, np.array(idx_1v)] += phi
            else: 
                self.beta[:, np.array(idx_1v)] += phi

        self.bound = np.sum(calculated_bounds)
        print(self.bound)
        
        self.last_bounds.append(self.bound)

    def get_beta(self, words, aspect):
        """ returns the topic-word distribution for a document with the respective topical content covariate (aspect)

        Args:
            words (ndarray): 1D-array with word indices for a specific document
            aspect (int, float32): topical content covariate for a specific document

        Raises:
            ValueError: _description_

        Returns:
            beta_doc_kv: topic-word distribution for a specific document, based on word indices and aspect
        """
        if not np.all((self.beta >= 0)): 
            raise ValueError("Some entries of beta are negative.")
        if self.interactions: 
            beta_doc_kv = self.beta[aspect][:,np.array(words)]
        else: 
            beta_doc_kv = self.beta[:,np.array(words)]
        return beta_doc_kv


    # _____________________________________________________________________
    def M_step(self):
        # Run M-Step 

        t1 = time.process_time()

        self.update_mu()
            # covar=self.settings['covariates']['X'],
            # enet=self.settings['gamma']['enet'],
            # ic_k=self.settings['gamma']['ic.k'],
            # maxits=self.settings['gamma']['maxits'],
            # mode=self.settings['gamma']['mode']


        self.update_sigma(
            nu=self.sigma, 
            sigprior=self.settings['sigma']['prior']
        )
        
        self.update_beta()

        print("Completed M-Step ({} seconds). \n".format(math.floor((time.process_time()-t1))))

    def update_mu(self):
        """
        updates the mean parameter for the logistic normal distribution
        """
        # Short hack
        self.mu = np.mean(self.eta, axis=0)

    def update_sigma(self, nu, sigprior):
        """
        Updates the variance covariance matrix for the logistic normal distribution

        Args:
            nu (_type_): variance-covariance for the variational document-topic distribution
            sigprior (_type_): prior for the var-cov. matrix for the log-normal
        """
        #find the covariance
        covariance = (self.eta - self.mu).T @ (self.eta-self.mu)
        sigma = (covariance + nu) / self.eta[0].shape[0] # Mayble replace with K-1
        self.sigma = np.diag(np.diag(sigma))*sigprior + (1-sigprior)*sigma

    def update_beta(self, kappa=None):
        """
        Updates the topic-word distribution

        Args:
            kappa (_type_, optional): topical content covariate. Defaults to None.
        """
        # computes the update for beta based on the SAGE model 
        # for now: just computes row normalized beta values
        if kappa is None:
            self.beta = self.beta/np.sum(self.beta, axis=1)[:,None]
        else: 
            print(f"implementation for {kappa} is missing")

    def expectation_maximization(self):
        t1 = time.process_time()
        for _iteration in range(100):
            self.E_step()
            self.M_step()

            if self.EM_is_converged():
                self.time_processed = time.process_time()-t1
                print(f'model converged in iteration {_iteration} after {self.time_processed}s')
                print('saving model...')
                self.save_model()
                break
            if self.max_its_reached(_iteration): 
                self.time_processed = time.process_time()-t1
                print(f'maximum number of iterations ({max_em_its}) reached after {self.time_processed} seconds')
                print('saving model...')
                self.save_model()
                break

    # _____________________________________________________________________    
    def EM_is_converged(self, convergence=None):
        new = self.bound
        old = self.last_bounds[-2]
        emtol = self.settings['convergence']['em.converge.thresh']

        convergence_check = np.abs((new - old)/np.abs(old))
        print(f'relative change: {convergence_check}')
        if convergence_check < emtol:
            return True
        else:
            return False
    
    def max_its_reached(self, _iteration):
        if _iteration == max_em_its-1:
            return True
        else:
            _iteration += 1
            return False

    def stable_softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        xshift = x-np.max(x)
        exps = np.exp(xshift)
        return exps / np.sum(exps)

    def softmax_weights(self, x, weight):
        """Compute weighted softmax values for each sets of scores in x.""" 
        xshift = x - np.max(x)
        exps = weight*np.exp(xshift)[:,None]
        return exps / np.sum(exps)
    
    def optimize_eta(self, eta, word_count, eta_extend, beta_doc, Ndoc, phi, theta): 
        """Optimizes the variational parameter eta given the likelihood and the gradient function
        """
        def f(eta, word_count, eta_extend, beta_doc, Ndoc, phi, theta):
            """objective for the variational update q(eta)
            """
            # precompute the difference since we use it twice
            diff = (eta-self.mu)
            #formula 
            part1 = np.sum(word_count * (eta_extend.max() + np.log(np.exp(eta_extend - eta_extend.max())@beta_doc)))-Ndoc*scipy.special.logsumexp(eta)
            part2 = -.5*diff.T@self.siginv@diff
            return np.float32(part2 + part1)*(-1)

        def df(eta, word_count, eta_extend, beta_doc, Ndoc, phi, theta):
            """gradient for the objective of the variational update q(etas)
            """
            #formula
            part1 = np.delete(np.sum(phi * word_count,axis=1) - Ndoc*theta, self.K-1)
            part2 = self.siginv@(eta-self.mu) # Check here!!! for dimensions
            return np.float32(part1 - part2)*(-1)
        
        # We want to maximize f, but numpy only implements minimize, so we
        # minimize -f
        
        return optimize.minimize(
            f, #TODO: negation of lhood
            x0=eta,
            args=(word_count, eta_extend, beta_doc, Ndoc, phi, theta),
            jac=df, #TODO: negation of grad
            method="BFGS"
            )

    def compute_hessian(self, eta, word_count, beta_doc_kv): 
        """
        computes hessian matrix for the variational distribution 
        first, off diagonal values are computed.
        diagonal values are replaced afterward
        in the end, hessian should be positive definite
        """
        # off diagonal entries
        eta_extend = np.insert(eta,len(eta),0)
        theta = self.stable_softmax(eta_extend)
        if not np.all((theta > 0) & (theta <= 1)): 
            raise ValueError("values of theta not between 0 and 1")
        expected_phi = self.softmax_weights(eta_extend, beta_doc_kv)
        p1_offdiag = (np.sqrt(word_count)*expected_phi)@(np.sqrt(word_count)*expected_phi).T
        # c.f. (theta * theta.t()); in the C++ implementation
        # gives a K by K matrix 
        p2_offdiag = sum(word_count)*theta[:,None]@theta[None,:]

        #should be positive
        neg_hess = p1_offdiag - p2_offdiag

        # diagonal entries
        p1_diag = np.sum(np.multiply(np.multiply(np.sqrt(word_count), expected_phi), np.sqrt(word_count)), axis = 1)
        p2_diag = sum(word_count)*theta

        #alter diagonal entries of the hessian
        np.fill_diagonal(neg_hess, np.diag(neg_hess)-p1_diag - p2_diag) 
        
        hess_kminus1bykminus1 = neg_hess[:-1,:-1]
        
        neg_hess = hess_kminus1bykminus1 + self.siginv # at this point, the hessian is complete

        return neg_hess
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
        dvec = np.where(dvec < magnitudes,magnitudes,dvec)
        # A Hermitian diagonally dominant matrix A with real non-negative diagonal entries is positive semidefinite. 
        np.fill_diagonal(M, dvec)
        return M
<<<<<<< HEAD
    def decompose_hessian(self, hess):
=======
        
    def invert_hessian(self, hess):
>>>>>>> cceec4b7e7f1da518b92fa95ef6286ad928870af
        """
        Decompose hessian via cholesky decomposition 
        error -> not properly converged: make the matrix positive definite
        np.linalg.cholesky(a) requires the matrix a to be hermitian positive-definite
        """
        try:  
            L = np.linalg.cholesky(hess)
        except:
            try:
<<<<<<< HEAD
                L = np.linalg.cholesky(self.make_pd(hess))
                print("converts Hessian via diagonal-dominance")
            except:
                L = np.linalg.cholesky(self.make_pd(hess) + 1e-5 * np.eye(hess.shape[0]))
                print("adds a small number to the hessian")
        
        return L
=======
                hess_inverse = np.linalg.cholesky(self.make_pd(hess))
                #print("converts Hessian via diagonal-dominance")
            except:
                hess_inverse = np.linalg.cholesky(self.make_pd(hess) + 1e-5 * np.eye(hess.shape[0]))
                #print("adds a small number to the hessian")
        return hess_inverse
>>>>>>> cceec4b7e7f1da518b92fa95ef6286ad928870af

    def optimize_nu(self, L): 
        """Given the inverse hessian returns the variance-covariance matrix for the variational distribution

        Args:
            L (np.array): lower triangular matrix of cholesky decomposition

        Returns:
            nu (np.array): variance-covariance matrix for the variational distribution q(eta|lambda, nu). 
        """
        nu = np.linalg.inv(np.triu(L))
        nu = nu@nu.T
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
            phi: update for the variational latent parameter z TODO: disentangle the update for z from bound calculation
        """
        eta_extend = np.insert(eta, len(eta), 0)
        exp_eta_extend = np.exp(eta_extend)
        theta = self.stable_softmax(eta_extend)
        #compute 1/2 the determinant from the cholesky decomposition
        detTerm =  -np.sum(np.log(L.diagonal()))
        diff = eta-mu
        ############## generate the bound and make it a scalar ##################
        beta_temp_kv = beta_doc_kv*exp_eta_extend[:,None]
        bound = np.log(theta[None:,]@beta_temp_kv)@word_count + detTerm - .5*diff.T@self.siginv@diff - self.sigmaentropy
        return bound
    
    def update_z(self, eta, beta_doc_kv): 
        """Compute the update for the variational latent parameter z

        Args:
            eta (np.array): 1D-array representing prior to the document-topic distribution
            beta_doc_kv (np.array): 2D-array (K by V) containing the topic-word distribution for a specific document

        Returns:
            phi: update for the variational latent parameter z
        """
        eta_extend = np.insert(eta, len(eta), 0)
        exp_eta_extend = np.exp(eta_extend)
        self.phi = beta_doc_kv*exp_eta_extend[:,None]
        return self.phi 
    
    def save_model(self):
        model = {
            'bound':self.last_bounds,
            'mu': self.mu,
            'sigma':self.sigma,
            'beta':self.beta,
            'settings': self.settings,
            'time_processed':self.time_processed}
        json.dump(model, open("stm_model.json", "w"), cls=NumpyEncoder)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
# %% Init params for training _____________________
# %%

# Parameter Settings (required for simulation process)
V=1000
num_topics = 30
A = 2
verbose = True
interactions = False #settings.kappa

# Initialization and Convergence Settings
init_type = "Random" #settings.init
ngroups = 1 #settings.ngroups
max_em_its = 30 #settings.convergence
emtol = 0.01 #settings.convergence
sigma_prior=0 #settings.sigma.prior


# Here we are simulating 100 documents with 100 words each. We are sampling from a multinomial distribution with dimension V.
# Note however that we will discard all elements from the vector V that do not occur.
# This leads to a dimension of the vocabulary << V
np.random.seed(123)

Corpus = CorpusCreation(n_topics=num_topics, n_docs=1000, n_words=150, V=5000, treatment=False, alpha='asymmetric')
Corpus.generate_documents()
betaindex = np.concatenate([np.repeat(0,len(Corpus.documents)/2), np.repeat(1,len(Corpus.documents)/2)])


# Set starting values and parameters
settings = {
    'dim':{
        'K': num_topics, #number of topics
        'V' : len(Corpus.dictionary), #number of words
        'A' : A, #dimension of topical content
        'N' : len(Corpus.documents),
    },
    'verbose':verbose,
    'kappa':{
        'interactions':interactions,
        'fixedintercept': True,
        'contrats': False,
        'mstep': {'tol':0.01, 'maxit':5}},
    'tau':{
        'mode': np.nan,
        'tol': 1e-5,
        'enet':1,
        'nlambda':250,
        'lambda.min.ratio':.001,
        'ic.k':2,
        'maxit':1e4},
    'init':{
        'mode':init_type, 
        'nits':20,
        'burnin':25,
        'alpha':50/num_topics,
        'eta':.01,
        's':.05,
        'p':3000},
    'convergence':{
        'max.em.its':max_em_its,
        'em.converge.thresh':emtol,
        'allow.neg.change':True,},
     'covariates':{
         'X':betaindex,
         'betaindex':betaindex,
    #     'yvarlevels':yvarlevels,
    #     'formula': prevalence,
    },
    'gamma':{
        'mode':'L1', #needs to be set for the m-step (update mu in the topical prevalence model)
        'prior':np.nan, #sigma in the topical prevalence model
        'enet':1, #regularization term
        'ic.k':2,#information criterion
        'maxits':1000,},
    'sigma':{
        'prior':sigma_prior,
        'ngroups':ngroups,},
}


# %%

model = STM(settings, Corpus.documents, Corpus.dictionary)


# %%

model.expectation_maximization()
