import numpy as np

import numpy.random as random
import math
from scipy import optimize
import scipy
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model


""" Class definition"""


class STM:
    """ Class for training STM."""
    def __init__(self, settings, documents, dictionary):
        self.Ndoc = None
        self.kappa_initialized = None
        self.eta = None
        self.eta_long = None
        self.siginv = None
        self.sigmaentropy = None
        self.settings=settings


        self.documents = documents
        self.dictionary = dictionary
        self.V = self.settings['dim']['V']
        self.K = self.settings['dim']['K']
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
        beta_init_normalized = (beta_init / np.sum(beta_init, axis=1)[:,None])
        if self.interactions: 
            self.beta = np.repeat(beta_init_normalized[None,:], self.A, axis=0)
            # test if probabilities sum to 1      
            [np.testing.assert_almost_equal(sum_over_words, 1) for i in range(self.A) for sum_over_words in np.sum(self.beta[i], axis=1)]
        else: 
            self.beta = beta_init_normalized
            [np.testing.assert_almost_equal(sum_over_words, 1) for i in range(self.A) for sum_over_words in np.sum(self.beta, axis=1)]
            


    def init_mu(self):
        self.mu = np.array([0]*(self.K-1))[:,None]

    def init_sigma(self):
        self.sigma = np.zeros(((self.K-1),(self.K-1)))
        np.fill_diagonal(self.sigma, 20)


    def init_lambda(self):
        """
        initialize lambda as a list to fill mean values for each document 
        dimension: N by K-1
        """
        self.lambd = [np.zeros(self.K-1)]*self.N
           

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

    
    def lhood(self, mu, eta, word_count, eta_long, beta_doc, Ndoc, phi, theta, neta):
        """ Computes Likelihood """
        # precompute the difference since we use it twice
        diff = (eta-mu)
        #formula 
        #-.5*diff@self.siginv@diff+np.sum(word_count * (eta_long.max() + np.log(np.exp(eta_long - eta_long.max())@beta_doc)))-Ndoc*scipy.special.logsumexp(eta)
        part1 = np.sum(word_count * (eta_long.max() + np.log(np.exp(eta_long - eta_long.max())@beta_doc)))-Ndoc*scipy.special.logsumexp(eta)
        part2 = -.5*diff@self.siginv@diff
        return part2 - part1
        
    def grad(self, mu, eta, word_count, eta_long, beta_doc, Ndoc, phi, theta, neta):
        """ Define Gradient """
        #formula
        part1 = np.delete(np.sum(phi * word_count,axis=1) - Ndoc*theta, neta)
        part2 = self.siginv@(eta-mu)
        return part1 - part2

    
    def e_step(self, documents):
        """ Optimize the following parameters: 
        Parameters: 
        - documents: a collection of document in the required (BoW) format
        - eta: a vector of length K-1 containing the initial starting value for eta
        - mu: a vector of length K-1 containing the prevalence prior
        - beta: a matrix containing the complete topic-word distribution for all content covariate levels
        - sigma: a k-1 by k-1 matrix containing the covariance matrix prevalence (MVN) prior
        - sigmainv: a K-1 by K-1 matrix containing the precision matrix of the MVN prior. 

        Return: 
        - phi: a K by V matrix containing the variational distribution for each token where V is the number of unique words in a the given document.
               They are in the order of appearance in the document. For words repeated more than once the sum of the column is the number of times that token appeared. 
        - lambda: A K-1 by 1 matrix containing the mean of the variational distribution for eta. This is actuallly just called eta in the output as it is also the point estimate. 
        - nu: A K-1 by K-1 matrix containing the covariance matrix of the variational distribution for eta. This is also the inverse Hessian matrix 
        - bound: The value of the document-level contribution to the global approximate evidence lower bound. 
        and returns the approximate evidence lower bound (ELBO)"""

        # 1) Initialize Sufficient Statistics 
        # The sufficient statistic of a set of independent identically distributed data observations is simply the sum of individual sufficient statistics.
        sigma_ss = np.zeros(((self.K-1),(self.K-1)))
        if self.interactions: 
            beta_ss = np.repeat(np.zeros((self.K,self.V)), self.A).reshape(self.A,self.K,self.V)
        else: 
            beta_ss = np.zeros((self.K,self.V))
        bound = np.repeat(0,self.N)
        
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

            #set document specs
            doc = documents[i]
            words_1v = self.get_words(doc)
            aspect = self.get_aspect(i)
            beta_doc_kv = self.get_beta(words_1v, aspect)
            #np.testing.assert_array_less(np.sum(beta_doc_kv, axis=1), 1)
            word_count_1v = np.array([y for x,y in doc]) #count of words in document
            Ndoc = np.sum(word_count_1v)
            # initial values
            theta_1k = stable_softmax(eta_long)
            np.testing.assert_equal(np.sum(theta_1k), 1)
            phi_vk = softmax_weights(eta_long, beta_doc_kv)
            #np.testing.assert_equal(np.sum(phi_vk), 1)

            
            # optimize variational posterior
            # does not matter if we use optimize.minimize(method='BFGS') or optimize fmin_bfgs()
            opti = optimize.minimize(self.lhood, x0=eta, args=(mu_i, word_count_1v, eta_long, beta_doc_kv, Ndoc, phi_vk, theta_1k, neta),
                            jac=self.grad, method="BFGS")
            
            # Compute Hessian, Phi and Lower Bound 
            hess = self.compute_hessian(eta = opti.x, word_count=word_count_1v, beta_doc_kv=beta_doc_kv)
            hess_inv = self.invert_hessian(hess)
            nu = self.compute_nu(hess_inv)
            bound_d, phi = self.lower_bound(hess_inv, mu = mu_i, word_count=word_count_1v, beta_doc_kv=beta_doc_kv, eta=opti.x)


            #solve hpb
            # doc_results = self.hpb(eta=opti.x,
            #                  word_count=word_count_1v,
            #                  mu=mu_i,
            #                  beta_doc_kv=beta_doc_kv)
            
            print(f"\nbound:{bound_d}")
            
            #Update sufficient statistics        
            sigma_ss = sigma_ss + nu
            if self.interactions: 
                beta_ss[aspect][:,np.array(words_1v)] = phi + np.take(beta_ss[aspect], words_1v, 1)
            else: 
                beta_ss[:,np.array(words_1v)] = phi + np.take(beta_ss, words_1v, 1)
            np.insert(bound, i, bound_d)
            self.lambd[i] = opti.x

        return sigma_ss, beta_ss, bound, nu

    def get_beta(self, words, aspect):
        """ returns the topic-word distribution for a document with the respective topical content covariate (aspect)"""
        if not np.all((self.beta >= 0)): 
            raise ValueError("Some entries of beta are negative.")
        if self.interactions: 
            beta_doc_kv = self.beta[aspect][:,np.array(words)]
        else: 
            beta_doc_kv = self.beta[:,np.array(words)]
        return beta_doc_kv

    def get_aspect(self, i):
        """returns the topical content covariate for document with index i"""
        aspect = self.betaindex[i]
        return aspect

    def get_words(self, doc):
        """ 
        returns the word indices for a given document
        """
        words = [x for x,y in doc]
        return words
    
    def compute_hessian(self, eta, word_count, beta_doc_kv): 
        """
        computes hessian matrix for the variational distribution 
        first, off diagonal values are computed.
        diagonal values are replaced afterward
        in the end, hessian should be positive definite
        """
        # off diagonal entries
        eta_long_K = np.insert(eta,len(eta),0)
        theta = stable_softmax(eta_long_K)
        if not np.all((theta > 0) & (theta < 1)): 
            raise ValueError("values of theta not between 0 and 1")
        expected_phi = softmax_weights(eta_long_K, beta_doc_kv)
        # in comparison to C++ implementation: EB = np.multiply(np.sqrt(word_count), expected_phi)
        # EB * EB.t() - sum(doc_cts) * (theta * theta.t());
        p1_offdiag = (np.sqrt(word_count)*expected_phi)@(np.sqrt(word_count)*expected_phi).T
        # c.f. (theta * theta.t()); in the C++ implementation
        # gives a K by K matrix 
        p2_offdiag = sum(word_count)*theta[:,None]@theta[None,:]

        #should be positive
        neg_hess = p1_offdiag - p2_offdiag

        # diagonal entries
        p1_diag = np.sum(np.multiply(np.multiply(np.sqrt(word_count), expected_phi), np.sqrt(word_count)), 1)
        p2_diag = sum(word_count)*theta

        #alter diagonal entries of the hessian
        np.fill_diagonal(neg_hess, np.diag(neg_hess)-p1_diag + p2_diag) 
        
        hess_kminus1bykminus1 = neg_hess[:-1,:-1]
        
        neg_hess = hess_kminus1bykminus1 + self.siginv # at this point, the hessian is complete

        return neg_hess

    def invert_hessian(self, hess):
        """
        Invert hessian via cholesky decomposition 
        error -> not properly converged: make the matrix positive definite
        np.linalg.cholesky(a) requires the matrix a to be hermitian positive-definite
        """
        try:  
            hess_inverse = np.linalg.cholesky(hess)
        except:
            #hess = self.validate_positive_definitive(hess)
            hess_inverse = np.linalg.cholesky(hess + 1e-12 * np.eye(hess.shape[0]))
        
        return hess_inverse

    def to_positive_definitive(self, M):
        M = np.matrix(M)
        M = (M + M.T) * 0.5
        k = 1
        I = np.eye(M.shape[0])
        w, v = np.linalg.eig(M)
        min_eig = v.min()
        M += (-min_eig * k * k + np.spacing(min_eig)) * I
        return M

    def validate_positive_definitive(self, M):   
        try:
            np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            M = self.to_positive_definitive(M)
        #Print the eigenvalues of the Matrix
        print(np.linalg.eigvalsh(M))
        return M
    
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
        #check if hermitian p.-d.
        #Print the eigenvalues of the Matrix
        print(np.linalg.eigvalsh(M))
        if not np.all(np.linalg.eigvals(M)>0):
            raise ValueError('The input matrix must be positive semidefinite')
        return M

    def compute_nu(self, hess_inverse): 
        """
        constructing nu
        """
        nu = np.linalg.inv(np.triu(hess_inverse))
        nu = nu@nu.T
        return nu
    
    def lower_bound(self, hess_inverse, mu, word_count, beta_doc_kv, eta):

        """
        computes the elbo as a scalar
        """
        eta_long_K = np.insert(eta,len(eta),0)
        expeta_K = np.exp(eta_long_K)
        theta = stable_softmax(eta_long_K)
        #compute 1/2 the determinant from the cholesky decomposition
        detTerm = -np.sum(np.log(hess_inverse.diagonal()))
        diff = eta-mu
        ############## generate the bound and make it a scalar ##################
        beta_temp_kv = beta_doc_kv*expeta_K[:,None]
        bound = np.log(theta[None:,]@beta_temp_kv)@word_count + detTerm - .5*diff.T@self.siginv@diff - self.sigmaentropy
        phi = beta_temp_kv
        return bound, phi


    """ Solve for Hessian/Phi/Bound returning the result"""
    def hpb(self, eta, word_count, mu, beta_doc_kv):
        eta_long_K = np.insert(eta,len(eta),0)
        # copy to mess with 
        # initial values
        theta = stable_softmax(eta_long_K) # 0 < theta < 1 ? 
        if not np.all((theta > 0) & (theta < 1)): 
            raise ValueError("values of theta not between 0 and 1")
        expeta_K = np.exp(eta_long_K)
        
        #column-wise multiplication of beta and expeta 
        beta_temp_kv = beta_doc_kv*expeta_K[:,None]
        
        beta_temp_kv_norm = np.divide(np.multiply(beta_temp_kv,np.sqrt(word_count)), np.sum(beta_temp_kv, axis=0))
        hess = beta_temp_kv_norm@beta_temp_kv_norm.T-np.sum(word_count)*(theta*theta.T) # hessian with shape KxK
        #we don't need beta_temp any more so we turn it into phi 
        #defined above in e-step: phi = softmax_weights(eta_long, beta_tuple)
        beta_temp_phi = np.multiply(beta_temp_kv_norm, np.sqrt(word_count)) # equals phi


        np.fill_diagonal(hess, np.diag(hess)-np.sum(beta_temp_phi, axis=1)-np.sum(word_count)*theta) #altered diagonal of h

        # drop last row and columns
        hess = np.delete(hess,eta.size,0)
        hess = np.delete(hess,eta.size,1)
        # if not np.all((hess >= 0) & (hess < 1)): 
        #     raise ValueError("values of hessian not between 0 and 1")
        hess = hess + self.siginv # at this point, the hessian is complete

        # Invert hessian via cholesky decomposition 
        # np.linalg.cholesky(hess)
        # error -> not properly converged: make the matrix positive definite
        #np.linalg.cholesky(a) requires the matrix a to be hermitian positive-definite
        self.make_pd(hess)
        #now we can do cholesky 
        nu = np.linalg.cholesky(hess)
        #compute 1/2 the determinant from the cholesky decomposition
        detTerm = -np.sum(np.log(nu.diagonal()))
        #Finish constructing nu
        nu = np.linalg.inv(np.triu(nu))
        nu = nu@nu.T
        # precompute the difference since we use it twice
        diff = eta-mu
        ############## generate the bound and make it a scalar ##################
        bound = np.log(theta[None:,]@beta_temp_kv)@word_count + detTerm - .5*diff.T@self.siginv@diff - self.sigmaentropy 
        ###################### return values as dictionary ######################
        phi = beta_temp_kv
        eta = {'lambd' : eta, 'nu':nu}
        
        result = {'phi':phi,'eta': eta,'bound': bound}
        
        return result

    def makeTopMatrix(self, x, data=None):
        return(data.loc[:,x]) # add intercept! 

    def opt_mu(self, covar, enet, ic_k, maxits, mode = "L1"):
        #prepare covariate matrix for modeling 
        try:
            covar = covar.astype('category')
        except:
            pass
        covar2D = np.array(covar)[:,None] #prepares 1D array for one-hot encoding (OHE) by making it 2D
        enc = OneHotEncoder(handle_unknown='ignore') #create OHE
        covarOHE = enc.fit_transform(covar2D).toarray() #fit OHE
        # TO-DO: mode = CTM if there are no covariates 
        # TO-DO: mode = Pooled if there are covariates requires variational linear regression with Half-Cauchy hyperprior
        # mode = L1 simplest method requires only glmnet (https://cran.r-project.org/web/packages/glmnet/index.html)
        if mode == "L1":
            linear_model = sklearn.linear_model.Lasso(alpha=enet)
            fitted_model = linear_model.fit(covarOHE,self.lambd)
        else: 
            raise ValueError('Updating the topical prevalence parameter requires a mode. Choose from "CTM", "Pooled" or "L1" (default).')
        gamma = np.insert(fitted_model.coef_, 0, fitted_model.intercept_).reshape(self.K-1,3)
        design_matrix = np.c_[ np.ones(covarOHE.shape[0]), covarOHE]
        #compute mu
        mu = design_matrix@gamma.T   
        return mu
        
    def opt_sigma(self, nu, mu, sigprior):
        #find the covariance
        # if ncol(mu) == 1: 
        #     covariance = np.cross(sweep(lambd, 2, STATS=as.numeric(mu), FUN="-")
        # else: 
        covariance = (self.lambd - mu).T@(self.lambd-mu)
        sigma = (covariance + nu)/self.lambd[0].shape[0]
        get_type(sigma)
        self.sigma = np.diag(np.diag(sigma))*sigprior + (1-sigprior)*sigma

    def opt_beta(self, beta_ss, kappa):
        #if its standard lda just row normalize
        if kappa is None: 
            norm_beta = beta_ss[[1]]/np.sum(beta_ss[[1]]) 
            self.beta = norm_beta
            #list(beta=list(beta_ss[[1]]/np.sum(beta_ss[[1]])))
        else: 
            print(f"implementation for {kappa} is missing")
        #if its a SAGE model (Eisenstein et al., 2013) use the distributed poissons
        # if settings['tau']['mode'] == "L1":
        #     out = mnreg(beta_ss, settings) 
        # else: 
        #     out = jeffreysKappa(beta_ss, kappa, settings)
        get_type(self.beta)

    def convergence_check(self, bound_ss, convergence, settings):
        verbose = settings['verbose']
        emtol = settings['convergence']['em.converge.thresh']
        maxits = settings['convergence']['max.em.its']
        # initialize the convergence object if empty
        if convergence is None: 
            convergence = {'bound':np.zeros(maxits), 'its':1, 'converged':False, 'stopits':False}
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

def stable_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    xshift = x-np.max(x)
    exps = np.exp(xshift)
    # assert sum(e_x/e_x.sum(axis=0)) == 1
    return exps / np.sum(exps)

def softmax_weights(x, weight):
    """Compute softmax values for each sets of scores in x.""" 
    xshift = x - np.max(x)
    exps = weight*np.exp(xshift)[:,None]
    return exps / np.sum(exps)

def get_type(x):
        """returns type of an object x"""
        msg = f'type of {x}: {type(x)}'
        return msg

