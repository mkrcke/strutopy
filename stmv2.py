# %%
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import json

# custom packages

from stm import STM
from simulate import generate_docs

import numpy as np

import numpy.random as random
import math
from scipy import optimize
import scipy
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model
import logging

logger = logging.getLogger(__name__)



# %%




class STM:

    def __init__(self, settings, documents, dictionary, dtype=np.float32):

        self.dtype = np.finfo(dtype).dtype # Why this?

        # Do we need this part?
        self.Ndoc = None
        self.kappa_initialized = None
        # self.eta = None
        self.eta_long = None
        self.siginv = None
        self.sigmaentropy = None

        # How to name this part?
        self.settings=settings
        self.documents = documents
        self.dictionary = dictionary

        # store user-supplied parameters
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
        
        self.last_bounds = [0.00001]

        self.init_global_params() # Think about naming. Are these global params?

    # _____________________________________________________________________
    def init_global_params(self):
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
        if self.interactions: 
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
        initialize lambda as a list to fill mean values for each document 
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
        
        Returns:
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

            eta_long = np.insert(self.eta[i], self.K-1, 0)

            #set document specs
            doc = documents[i] # TODO: Make documents a numpy array

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
            theta_1k = self.stable_softmax(eta_long)

            phi_vk = self.softmax_weights(eta_long, beta_doc_kv)
            
            # optimize variational posterior
            # does not matter if we use optimize.minimize(method='BFGS') or optimize fmin_bfgs()

            # x0 : ndarray, shape (n,)
            #     Initial guess. Array of real elements of size (n,),
            #     where ``n`` is the number of independent variables.
            # args : tuple, optional


            eta_hat_i = optimize.minimize(
                self.lhood,
                x0=self.eta[i],
                args=(self.mu, word_count_1v, eta_long, beta_doc_kv, Ndoc, phi_vk, theta_1k, self.K-1),
                jac=self.grad,
                method="BFGS"
            )

            self.eta[i] = eta_hat_i.x
            
            # Compute Hessian, Phi and Lower Bound 

            hess_inv_i = eta_hat_i.hess_inv # TODO: Make a self.inverted_hessian[i]
            # print(hess_inv_i.message)
            #hess = self.compute_hessian(eta = opti.x, word_count=word_count_1v, beta_doc_kv=beta_doc_kv)
            #hess_inv = self.invert_hessian(hess_inv)

            # Delta NU
            nu = self.compute_nu(hess_inv_i)

            # Delta Bound, Delta Phi
            bound_i, phi = self.lower_bound(
                hess_inv_i,
                mu=self.mu,
                word_count=word_count_1v,
                beta_doc_kv=beta_doc_kv,
                eta=eta_hat_i.x
            )
            #print(bound_i)

            calculated_bounds.append(bound_i)

            self.sigma += nu

            if self.interactions:
                self.beta[aspect][:, np.array(idx_1v)] += phi
            else: 
                self.beta[:, np.array(idx_1v)] += phi

        self.bound = np.sum(calculated_bounds)
        print(self.bound)
        
        self.last_bounds.append(self.bound)

    def get_beta(self, words, aspect):
        """ returns the topic-word distribution for a document with the respective topical content covariate (aspect)"""
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

        self.opt_mu()
            # covar=self.settings['covariates']['X'],
            # enet=self.settings['gamma']['enet'],
            # ic_k=self.settings['gamma']['ic.k'],
            # maxits=self.settings['gamma']['maxits'],
            # mode=self.settings['gamma']['mode']


        self.opt_sigma(
            nu=self.sigma, 
            sigprior=self.settings['sigma']['prior']
        )
        
        self.opt_beta()

        print("Completed M-Step ({} seconds). \n".format(math.floor((time.process_time()-t1))))

    def opt_mu(self):
        # Short hack
        self.mu = np.mean(self.eta, axis=0)

    def opt_sigma(self, nu, sigprior):
        #find the covariance
        covariance = (self.eta - self.mu).T @ (self.eta-self.mu)
        sigma = (covariance + nu) / self.eta[0].shape[0] # Mayble replace with K-1
        self.sigma = np.diag(np.diag(sigma))*sigprior + (1-sigprior)*sigma

    def opt_beta(self, kappa=None):
        # computes the update for beta based on the SAGE model 
        # for now: just computes row normalized beta values
        if kappa is None:
            self.beta = self.beta/np.sum(self.beta, axis=1)[:,None]
        else: 
            print(f"implementation for {kappa} is missing")

    def inference(self):

        for _ in range(100):
            self.E_step()
            self.M_step()

            if self.EM_is_converged():
                break

    # _____________________________________________________________________    
    def EM_is_converged(self, convergence=None):
        new = self.bound
        old = self.last_bounds[-2]
        emtol = self.settings['convergence']['em.converge.thresh']

        convergence_check = (new - old)/np.abs(old)

        if convergence_check < emtol:
            return True
        else:
            return False

    def convergence_check(self, convergence):
        verbose = self.settings['verbose']
        emtol = self.settings['convergence']['em.converge.thresh']
        maxits = self.settings['convergence']['max.em.its']
        # initialize the convergence object if empty
        if convergence is None: 
            convergence = {'bound':np.zeros(maxits), 'its':0, 'converged':False, 'stopits':False}
        # fill in the current bound
        convergence['bound'][convergence.get('its')] = self.bound
        # if not first iteration
        if convergence['its']>0:
            old = convergence['bound'][convergence['its']-1] #assign bound from previous iteration
            new = convergence['bound'][convergence['its']]
            convergence_check = (new-old)/np.abs(old)
            if emtol!=0: 
                if convergence_check>0 | self.settings['convergence']['allow.neg.change']:
                    if convergence_check < emtol: 
                        convergence['converged'] = True
                        convergence['stopits'] = True
                        if verbose: 
                            print('Model converged.')
                            return convergence
        if convergence['its']+1==maxits: 
            if (verbose) & (emtol != 0): 
                print('Model terminated before convergence reached.\n')
            if (verbose) & (emtol == 0): 
                print('Model terminated after requested number of steps. \n')
            convergence['stopits'] = True
            return convergence
        convergence['its'] += 1
        return convergence['converged']

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

    def get_type(self, x):
        """returns type of an object x"""
        msg = f'type of {x}: {type(x)}'
        return msg

    def lhood(self, mu, eta, word_count, eta_long, beta_doc, Ndoc, phi, theta, neta):
        """
        Computes Likelihood
        """
        # precompute the difference since we use it twice
        diff = (eta-mu)
        #formula 
        #-.5*diff@self.siginv@diff+np.sum(word_count * (eta_long.max() + np.log(np.exp(eta_long - eta_long.max())@beta_doc)))-Ndoc*scipy.special.logsumexp(eta)
        part1 = np.sum(word_count * (eta_long.max() + np.log(np.exp(eta_long - eta_long.max())@beta_doc)))-Ndoc*scipy.special.logsumexp(eta)
        part2 = .5*diff@self.siginv@diff.T
        return np.float32(part2 - part1)

    def grad(self, mu, eta, word_count, eta_long, beta_doc, Ndoc, phi, theta, neta):
        """
        Define Gradient
        """
        #formula
        part1 = np.delete(np.sum(phi * word_count,axis=1) - Ndoc*theta, neta)
        part2 = self.siginv@(eta-mu).T # Check here!!! for dimensions
        return part2 - part1

    def compute_nu(self, hess_inverse): 
        """
        constructing nu
        """
        nu = np.linalg.inv(np.triu(hess_inverse))
        nu = nu@nu.T
        return nu

    def lower_bound(self, hess_inverse, mu, word_count, beta_doc_kv, eta):
        """
        computes the ELBO for each document
        """
        eta_long_K = np.insert(eta, len(eta), 0)
        expeta_K = np.exp(eta_long_K)
        theta = self.stable_softmax(eta_long_K)
        #compute 1/2 the determinant from the cholesky decomposition
        detTerm = -np.sum(np.log(hess_inverse.diagonal()))
        diff = eta-mu
        ############## generate the bound and make it a scalar ##################
        beta_temp_kv = beta_doc_kv*expeta_K[:,None]
        bound = np.log(theta[None:,]@beta_temp_kv)@word_count + detTerm - .5*diff.T@self.siginv@diff - self.sigmaentropy
        phi = beta_temp_kv
        return bound, phi



# %% Init params for training _____________________
# %%

# Parameter Settings (required for simulation process)
V=500
num_topics = 3
A = 2
verbose = True
interactions = False #settings.kappa

# Initialization and Convergence Settings
init_type = "Random" #settings.init
ngroups = 1 #settings.ngroups
max_em_its = 20 #settings.convergence
emtol = 1e-5 #settings.convergence
sigma_prior=0 #settings.sigma.prior


def basic_simulations(n_docs, n_words, V, ATE, alpha, display=True):
    generator = generate_docs(n_docs, n_words, V, ATE, alpha)
    documents = generator.generate(n_docs)
    if display == True:
        generator.display_props()
    return documents

# Here we are simulating 100 documents with 100 words each. We are sampling from a multinomial distribution with dimension V.
# Note however that we will discard all elements from the vector V that do not occur.
# This leads to a dimension of the vocabulary << V
np.random.seed(123)
documents, vocabulary = basic_simulations(n_docs=100, n_words=40, V=500, ATE=.2, alpha=np.array([.3,.4,.3]), display=False)
betaindex = np.concatenate([np.repeat(0,50), np.repeat(1,50)])
num_topics = 3
dictionary=np.arange(vocabulary)

# Set starting values and parameters
settings = {
    'dim':{
        'K': num_topics, #number of topics
        'V' : vocabulary, #number of words
        'A' : A, #dimension of topical content
        'N' : len(documents),
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

model = STM(settings, documents, dictionary)


# %%

model.inference()