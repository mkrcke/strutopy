
import numpy as np 
from scipy.sparse import diags, csr_matrix
from sklearn import linear_model

def mnreg(self, beta_ss): 
    """estimation of distributed poisson regression for the update of the kappa parameters

    @param: beta_ss (np.ndarray) estimated word-topic distribution of the current EM-iteration with dimension K x V
    """
    
    contrast = False
    A = self.A
    K = self.K
    interact = True
    fixed_intercept = True
    alpha = 1e-10
    maxit=1e-4
    nlambda = 250 
    ic_k = 2
    thresh = 1e-5
    tol=1e-5
    enet=1
    nlambda=250
    lamda_min_ratio=.001

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
        vecj = np.concatenate((vecj, np.tile(np.arange(self.K+self.A+1, self.K+self.A+(counts.shape[0])))))
        if interact: 
            veci = np.concatenate((veci, np.arange(0,counts.shape[0])))
            vecj = np.concatenate((vecj, np.arange(self.K+self.A+1, self.K+self.A+counts.shape[0])))
        vecv = np.ones(len(veci))
        covar = csr_matrix((vecv, (veci,vecj))) 

        if fixed_intercept: 
            m = self.wcounts
            m = np.log(m) - np.log(np.sum(m))
        else: 
            m = 0 
        
        mult_nobs = counts.sum(axis=1)  
        offset = np.log(mult_nobs)
        counts = np.split(self.beta, self.beta.shape[1], axis=1)


    ############################
    ### Distributed Poissons ###
    ############################
    out = []
    #now iterate over the vocabulary
    for i in range(len(counts)):
        if m == 0: 
            offset2 = offset
        else: 
            offset2 = m[i] + offset
        mod = None
        #while mod is None: 
        clf = linear_model.PoissonRegressor(fit_intercept=offset2, maxiter=maxit, tol=tol, alpha=alpha)
        mod = clf.fit(covar, counts[[i]])
            #if it didn't converge, increase nlambda paths by 20% 
            # if(is.null(mod)) nlambda <- nlambda + floor(.2*nlambda)
        print(f'Estimated coefficients for word {i}.')
        print(mod.coef_)
        coef = mod.coef_
        out.append(coef)

    if not fixed_intercept: 
        m = out[0]
        coef = out[1:]
    
    kappa = np.split(coef, coef.shape[0])

    ###################
    ### predictions ###
    ###################

    linpred = covar@coef
     #linpred = m + linpred  
    
    explinpred = np.exp(linpred)
    beta =  explinpred/np.sum(explinpred, axis=1)

    return beta, kappa


        
    

    




