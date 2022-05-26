from doctest import DONT_ACCEPT_TRUE_FOR_1
import numpy as np

import numpy.random as random
import math
from scipy import optimize
import scipy
from sklearn.preprocessing import OneHotEncoder
import sklearn

""" Class definition"""


class STM:
    """ Class for training STM."""
    def __init__(self, settings, documents, dictionary, K = 100):
        self.Ndoc = None
        self.kappa_initialized = None
        self.eta = None
        self.eta_long = None
        self.siginv = None
        self.doc_ct = None
        self.beta_tuple = None
        self.theta = None
        self.neta = None
        self.settings=settings


        self.documents = documents
        self.dictionary = dictionary

        self.K = K
        self.V = len(self.dictionary)
        self.A = self.settings['dim']['A'] # TODO: when data changes ...
        self.N = len(self.documents)
        self.interactions = settings['kappa']['interactions']
        self.betaindex = settings['covariates']['betaindex']
        
        # Set global params
        self.init_beta() 
        self.init_mu()
        self.init_lambda()
        self.init_sigma()
        self.init_kappa()

    def init_beta(self):
        beta_init = random.gamma(.1,1, self.V*self.K).reshape(self.K,self.V)
        beta_init_normalized = (beta_init / beta_init.sum(axis=1)[:,None])
        self.beta = np.repeat(beta_init_normalized, self.A).reshape(self.A, self.K, self.V)

    def init_mu(self):
        self.mu = np.array([0]*(self.K-1))[:,None]

    def init_sigma(self):
        self.sigma = np.zeros(((self.K-1),(self.K-1)))
        np.fill_diagonal(self.sigma, 20)


    def init_lambda(self):
        self.lambd = np.zeros((self.N, (self.K-1)))
           

    """ Initializing Topical Content Model Parameters"""
    def init_kappa(self): 
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

    """ Compute Likelihood Function """
    def lhood(self, mu, eta, word_count, eta_long, beta_tuple, Ndoc, phi, theta, neta):
        #formula 
        #rewrite LSE to prevent overflow
        part1 = np.sum(word_count * (eta_long.max() + np.log(np.exp(eta_long - eta_long.max())@beta_tuple)))-Ndoc*scipy.special.logsumexp(eta)
        part2 = .5*(eta-mu)@siginv@(eta-mu)
        return - (part2 + part1)
        
    """ Define Gradient """
    def grad(self):
        #formula
        part1 = np.delete(np.sum(self.phi * self.doc_ct,axis=1) - np.sum(self.doc_ct)*self.theta, self.neta)
        part2 = self.siginv@(self.eta-self.mu)
        return part2 - part1

    """ Optimize Parameter Space """
    def e_step(self, documents):

        # 1) Initialize Sufficient Statistics 
        sigma_ss = np.zeros(((self.K-1),(self.K-1)))
        beta_ss_i = np.zeros((self.K,self.V))
        beta_ss = np.repeat(beta_ss_i, self.A).reshape(self.A,self.K,self.V)
        bound = np.repeat(0,self.N)
        lambd = np.repeat(0,self.N)
        
        # 2) Precalculate common components
        while True:
            try: 
                sigobj = np.linalg.cholesky(self.sigma) #initialization of sigma not positive definite
                sigmaentropy = np.sum(np.log(np.diag(sigobj)))
                siginv = np.linalg.inv(sigobj).T*np.linalg.inv(sigobj)
                break
            except:
                print("Cholesky Decomposition failed, because sigma is not positive definite!")
                sigmaentropy = .5*np.linalg.slogdet(self.sigma)[1]
                siginv = np.linalg.solve(self.sigma)           
            
        # 3) Document Scheduling
        # For right now we are just doing everything in serial.
        # the challenge with multicore is efficient scheduling while
        # maintaining a small dimension for the sufficient statistics.
        ############
        # input checks
        # get mu from dict for second iteration  
        if type(self.mu) is dict: 
            self.mu = self.mu.get('mu')
            update_mu = True

        else:
            mu_i = self.mu.flatten()
            update_mu = False
        

        #set parameters for one document (i)
        for i in range(self.N):

            if update_mu: 
                mu_i = self.mu[i]
            
            eta = self.lambd[i]
            neta = len(eta)
       
            eta_long = np.insert(eta,neta,0)

            doc = documents[i]
            words = [x for x,y in doc]
            aspect = self.betaindex[i]
            beta_tuple = self.beta[aspect][:,np.array(words)]

            #set document specs
            word_count = np.array([y for x,y in doc]) #count of words in document
            Ndoc = np.sum(word_count)
            # initial values
            theta = softmax(eta_long)
            phi = softmax_weights(eta_long, beta_tuple)
            # optimize variational posterior
            result = optimize.fmin_bfgs(self.lhood, x0=eta, args=(mu_i, eta, word_count, eta_long, beta_tuple, Ndoc, phi, theta, neta),
                            fprime=self.grad)
            #solve hpb
            doc_results = self.hpb(eta=result,
                            doc_ct=word_count,
                            mu=mu_i,
                            siginv=siginv,
                            beta_tuple=beta_tuple,
                            sigmaentropy=sigmaentropy,
                            theta=theta)
            
            print(f"\nbound:{doc_results['bound']}")
            print(f"\nresults:{doc_results}")
            
            #Update sufficient statistics        
            sigma_ss = sigma_ss + doc_results['eta'].get('nu')
            beta_ss[aspect][:,np.array(words)] = doc_results.get('phi') + np.take(beta_ss[aspect], words, 1)
            bound[i] = doc_results['bound']
            lambd[i] = doc_results['eta'].get('lambd')

        return sigma_ss, beta_ss, bound, lambd


    """ Solve for Hessian/Phi/Bound returning the result"""
    def hpb(self, eta, doc_ct, mu, siginv, beta_tuple, sigmaentropy, theta):
        eta_long = np.insert(eta,len(eta),0)
        # copy to mess with 
        beta_temp = beta_tuple
        #column-wise multiplication of beta and expeta (!) TO-DO: not eta_long! 
        expeta = np.exp(eta_long)
        # beta_temp = beta_temp*expeta[:,None]
        beta_temp = beta_temp*expeta[:,None]
        
        beta_temp = (np.sqrt(doc_ct)[:,None] / np.sum(beta_temp, axis=0)[:,None]) * beta_temp.T # with shape (VxK)
        hess = beta_temp.T@beta_temp-np.sum(doc_ct)*(theta*theta.T) # hessian with shape KxK
        #we don't need beta_temp any more so we turn it into phi 
        beta_temp = beta_temp.T * np.sqrt(doc_ct) # should equal phi ?! 

        np.fill_diagonal(hess, np.diag(hess)-np.sum(beta_temp, axis=1)-np.sum(doc_ct)*theta) #altered diagonal of h
        
        # drop last row and columns
        hess = np.delete(hess,eta.size,0)
        hess = np.delete(hess,eta.size,1)
        hess = hess + siginv # at this point, the hessian is complete

        # Invert hessian via cholesky decomposition 
        #np.linalg.cholesky(hess)
        # error -> not properly converged: make the matrix positive definite
        
        #def make_pd(): """Convert matrix X to be positive definite."""
        dvec = hess.diagonal()
        magnitudes = sum(abs(hess), 1) - abs(dvec)
        # cholesky decomposition works only for symmetric and positive definite matrices
        dvec = np.where(dvec < magnitudes, magnitudes, dvec)
        # A Hermitian diagonally dominant matrix A with real non-negative diagonal entries is positive semidefinite. 
        np.fill_diagonal(hess, dvec)
        #that was sufficient to ensure positive definiteness so no we can do cholesky 
        nu = np.linalg.cholesky(hess)
        #compute 1/2 the determinant from the cholesky decomposition
        detTerm = -np.sum(np.log(nu.diagonal()))
        #Finish constructing nu
        nu = np.linalg.inv(np.triu(nu))
        nu = nu@nu.T
        # precompute the difference since we use it twice
        diff = eta-mu
        ############## generate the bound and make it a scalar ##################
        bound = np.log(theta[None:,]@beta_temp)@doc_ct + detTerm - .5*diff.T@siginv@diff - sigmaentropy 
        ###################### return values as dictionary ######################
        phi = beta_temp
        eta = {'lambd' : eta, 'nu':nu}
        
        result = {'phi':phi,'eta': eta,'bound': bound}
        
        return result

    def makeTopMatrix(self, x, data=None):
        return(data.loc[:,x]) # add intercept! 

    def opt_mu(self, lambd, covar, enet, ic_k, maxits, mode = "L1"):
        #prepare covariate matrix for modeling 
        covar = covar.astype('category')
        covar2D = np.array(covar)[:,None] #prepares 1D array for one-hot encoding by making it 2D
        enc = OneHotEncoder(handle_unknown='ignore') #create OHE
        covarOHE = enc.fit_transform(covar2D).toarray() #fit OHE
        # TO-DO: mode = CTM if there are no covariates 
        # TO-DO: mode = Pooled if there are covariates requires variational linear regression with Half-Cauchy hyperprior
        # mode = L1 simplest method requires only glmnet (https://cran.r-project.org/web/packages/glmnet/index.html)
        if mode == "L1":
            model = sklearn.linear_model.Lasso(alpha=enet)
            fitted_model = model.fit(covarOHE,lambd)
        else: 
            raise ValueError('Updating the topical prevalence parameter requires a mode. Choose from "CTM", "Pooled" or "L1" (default).')
        gamma = np.insert(fitted_model.coef_, 0, fitted_model.intercept_).reshape(9,3)
        design_matrix = np.c_[ np.ones(covarOHE.shape[0]), covarOHE]
        #compute mu
        mu = design_matrix@gamma.T   
        return {
            'mu':mu,
            'gamma':gamma
            }
        
    def opt_sigma(self, nu, lambd, mu, sigprior):
        #find the covariance
        # if ncol(mu) == 1: 
        #     covariance = np.cross(sweep(lambd, 2, STATS=as.numeric(mu), FUN="-")
        # else: 
        covariance = (lambd - mu).T@(lambd-mu)
        sigma = (covariance + nu)/lambd.shape[1]
        get_type(sigma)
        self.sigma = np.diag(np.diag(sigma))*sigprior + (1-sigprior)*sigma

    def opt_beta(self, beta_ss, kappa):
        #if its standard lda just row normalize
        if kappa is None: 
            norm_beta = beta_ss[[1]]/np.sum(beta_ss[[1]]) 
            beta = {'beta': norm_beta}
            #list(beta=list(beta_ss[[1]]/np.sum(beta_ss[[1]])))
        else: 
            print(f"implementation for {kappa} is missing")
        #if its a SAGE model (Eisenstein et al., 2013) use the distributed poissons
        # if settings['tau']['mode'] == "L1":
        #     out = mnreg(beta_ss, settings) 
        # else: 
        #     out = jeffreysKappa(beta_ss, kappa, settings)
        get_type(beta)
        self.beta = beta

    def convergence_check(self, bound_ss, convergence, settings):
        verbose = settings['verbose']
        emtol = settings['convergence']['em.converge.thresh']
        maxits = settings['convergence']['max.em.its']
        # initialize the convergence object if empty
        if convergence is None: 
            convergence = {'bound':np.zeros(max_em_its), 'its':1, 'converged':False, 'stopits':False}
        # fill in the current bound
        convergence['bound'][convergence.get('its')] = np.sum(bound_ss)
        # if not first iteration
        if convergence['its']>1:
            old = convergence['bound'][convergence.get(its)-1] #assign bound from previous iteration
            new = convergence['bound'][convergence.get(its)]
            convergence_check = (new-old)/np.abs(old)
            if emtol!=0: 
                if convergence_check>0 | settings['convergence']['allow.neg.change']:
                    if convergence_check < emtol: 
                        convergence['converged'] = True
                        convergence['stopits'] = True
                        if verbose: 
                            print('Model converged.')
                            return convergence
        if convergence['its']==maxits: 
            if verbose & emtol != 0: 
                print('Model terminated before convergence reached.\n')
            if verbose & emtol == 0: 
                print('Model terminated after requested number of steps. \n')
            convergence['stopits'] = True
            return convergence
        convergence['its'] += 1
        return convergence

    """ Useful functions """

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def softmax_weights(x, weight):
    """Compute softmax values for each sets of scores in x."""
    e_x = weight*np.exp(x - np.max(x))[:,None]
    return e_x / e_x.sum(axis=0)

def get_type(x):
        """returns type of an object x"""
        msg = f'type of {x}: {type(x)}'
        return msg

