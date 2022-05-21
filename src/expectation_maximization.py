import numpy as np
import time
import pandas as pd
import numpy.random as random
import matplotlib.pyplot as plt
import math
from gensim import corpora
from scipy import optimize
import scipy
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import sklearn


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


""" Initializing Global Model Parameters"""
def init_stm(documents, settings): 
      
    K = settings['dim']['K']
    V = settings['dim']['V']
    A = settings['dim']['A']
    N = settings['dim']['N']
    
    #Random initialization, TO-DO: Improve initialization 
    mu = np.array([0]*(K-1))[:,None]
    sigma = np.zeros(((K-1),(K-1)))
    diag = np.diagonal(sigma, 0)
    diag.setflags(write=True)
    diag.fill(20)
    beta = random.gamma(.1,1, V*K).reshape(K,V)
    beta = (beta / beta.sum(axis=1)[:,None])
    lambd = np.zeros((N, (K-1)))
    
    #turn beta into a list and assign it for each aspect
    beta = np.repeat(beta,A).reshape(A,K,V) 
    kappa_initialized = init_kappa(documents, K, V, A, interactions=settings['kappa']['interactions'])
      
    #create model object
    model = {'mu':mu, 'sigma':sigma, 'beta': beta, 'lambda': lambd, 'kappa':kappa_initialized}
    
    return(model)

""" Initializing Topical Content Model Parameters"""
def init_kappa(documents, K, V, A, interactions): 
    # read in documents and vocab
    flat_documents = [item for sublist in documents for item in sublist]
    m = []

    total_sum = sum(n for _, n in flat_documents)

    for elem in flat_documents: 
        m.append(elem[1] / total_sum)

    m = np.log(m) - np.log(np.mean(m)) #logit of m


    #Defining parameters
    aspectmod = A > 1 # if there is more than one level for the topical content
    if(aspectmod):
        interact = interactions # allow for the choice to interact
    else:
        interact = FALSE

    #Create the parameters object
    parLength = K + A * aspectmod + (K*A)*interact

    #create covariates. one element per item in parameter list.
    #generation by type because its conceptually simpler
    if not aspectmod & interact:
        covar = {'k': np.arange(K),
             'a': np.repeat(np.nan, parLength), #why parLength? 
             'type': np.repeat(1, K)}

    if(aspectmod & interact == False):
        covar = {'k': np.append(np.arange(K), np.repeat(np.nan, A)),
                 'a': np.append(np.repeat(np.nan, K), np.arange(A)), 
                 'type': np.append(np.repeat(1, K), np.repeat(2, A))}      
    if(interact):
        covar = {'k': np.append(np.arange(K), np.append(np.repeat(np.nan, A), np.repeat(np.arange(K), A))),
                 'a': np.append(np.repeat(np.nan, K), np.append(np.arange(A), np.repeat(np.arange(A), K))), 
                 'type': np.append(np.repeat(1, K), np.append(np.repeat(2, A),  np.repeat(3,K*A)))}

    kappa = {'out': {'m':m,
                     'params' : np.tile(np.repeat(0,V), (parLength, 1)),
                     'covar' : covar
                     #'kappasum':, why rolling sum?
                    }
            }

    return(kappa['out'])

""" Compute Likelihood Function """
def lhood(eta, mu, siginv, doc_ct, Ndoc, eta_long, beta_tuple, phi, theta, neta):
    
    #formula 
    #rewrite LSE to prevent overflow
    part1 = np.sum(doc_ct * (eta_long.max() + np.log(np.exp(eta_long - eta_long.max())@beta_tuple)))-np.sum(doc_ct)*scipy.special.logsumexp(eta)
    part2 = .5*(eta-mu)@siginv@(eta-mu)
    
    out = part2 - part1
    
    return -out

""" Define Gradient """
def grad(eta, mu, siginv, doc_ct,  Ndoc, eta_long, beta_tuple, phi, theta, neta):

    #formula
    part1 = np.delete(np.sum(phi * doc_ct,axis=1) - np.sum(doc_ct)*theta, neta)
    part2 = siginv@(eta-mu)

    return part2 - part1

""" Optimize Parameter Space """
def e_step(documents, mu, sigma, lambd, beta):
    #quickly define useful constants
    V = beta['beta'][0].shape[1] # ncol
    K = beta['beta'][0].shape[0] # nrow
    N = len(documents)
    A = len(beta['beta'])
    
    # 1) Initialize Sufficient Statistics 
    sigma_ss = np.zeros(((K-1),(K-1)))
    beta_ss_i = np.zeros((K,V))
    beta_ss = np.repeat(beta_ss_i, A).reshape(A,K,V)
    bound = np.repeat(0,N)
    #lambd = np.repeat(0,N)
    
    # 2) Precalculate common components
    sigobj = np.linalg.cholesky(sigma) #initialization of sigma not positive definite
    sigmaentropy = np.sum(np.log(np.diag(sigobj)))
    siginv = np.linalg.inv(sigobj).T*np.linalg.inv(sigobj)
    
    # 3) Document Scheduling
    # For right now we are just doing everything in serial.
    # the challenge with multicore is efficient scheduling while
    # maintaining a small dimension for the sufficient statistics.
    ############
    # input checks
    # get mu from dict for second iteration  
    if type(mu) is dict: 
        mu = mu.get('mu')
        update_mu = True

    else:
        mu_i = mu.flatten()
        update_mu = False
    

    #set parameters for one document (i)
    for i in range(N):

        if update_mu: 
            mu_i = mu[i]
        
        eta=lambd[i]
        neta = len(eta)
        eta_long = np.insert(eta,len(eta),0)

        doc = documents[i]
        words = [x for x,y in doc]
        aspect = betaindex.iloc[i]
        #beta_i = beta['beta'][aspect][:,[words]] # replace with beta_ss[aspect][:,np.array(words)]
        beta_tuple = beta['beta'][aspect][:,np.array(words)]

        #set document specs
        doc_ct = np.array([y for x,y in doc]) #count of words in document
        Ndoc = np.sum(doc_ct)

        # initial values
        #beta_tuple = beta_i.reshape(K,beta_i.shape[2])
        theta = softmax(eta_long)
        phi = softmax_weights(eta_long, beta_tuple)
        # optimize variational posterior
        result = optimize.fmin_bfgs(lhood,x0=eta,
                           args=(mu_i, siginv, Ndoc, doc_ct, eta_long, beta_tuple, phi, theta, neta),
                           fprime=grad)
        #solve hpb
        doc_results = hpb(eta=result,
                          doc_ct=doc_ct,
                          mu=mu_i,
                          siginv=siginv,
                          beta_tuple=beta_tuple,
                          sigmaentropy=sigmaentropy,
                          theta=theta)
        
        #3) Update sufficient statistics        
        #print(f"Input:eta: {doc_results['eta'].get('nu').shape}\nphi:{doc_results['phi'].shape}")
        print(f"\nbound:{doc_results['bound']}")
        print(f"\nresults:{doc_results}")
        sigma_ss = sigma_ss + doc_results['eta'].get('nu')

        #beta_ss[aspect][:,[words]] = beta_ss[aspect][:,[words]].reshape(K,beta_ss[aspect][:,[words]].shape[2])         
        beta_ss[aspect][:,np.array(words)] = doc_results.get('phi') + np.take(beta_ss[aspect], words, 1)
        bound[i] = doc_results['bound']
        lambd[i] = doc_results['eta'].get('lambd')
        #4) Combine and Return Sufficient Statistics
        results = {'sigma':sigma_ss, 'beta':beta_ss, 'bound': bound, 'lambd': lambd}

    return results

""" Solve for Hessian/Phi/Bound returning the result"""
def hpb(eta, doc_ct, mu, siginv, beta_tuple, sigmaentropy, theta):
    eta_long = np.insert(eta,len(eta),0)
    # copy to mess with 
    beta_temp = beta_tuple
    #column-wise multiplication of beta and expeta (!) TO-DO: not eta_long! 
    # expeta = np.exp(eta_long)
    # beta_temp = beta_temp*expeta[:,None]
    beta_temp = beta_temp*eta_long[:,None]
    
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

def makeTopMatrix(x, data=None):
    return(data.loc[:,x]) # add intercept! 

def stm_control(documents, vocab, settings, model=None):
    ##########
    #Step 1: Initialize Parameters
    ##########
    #TO-DO: Optimize with ngroups
    if model == None:
        print('Call init_stm()')
        model = init_stm(documents, settings) #initialize
    else: 
        model = model
    # unpack initialized model
    mu = model['mu']
    sigma = model['sigma']
    lambd = model['lambda'] 
    beta = {'beta': model['beta'],
            'kappa': model['kappa']}
    convergence = None
    #discard the old object
    del model
    betaindex = settings['covariates']['betaindex']
    #Pull out some book keeping elements
    betaindex = settings['covariates']['betaindex']
    ############
    #Step 2: Run EM
    ############
    t1 = time.process_time()
    suffstats = [] 
    stopits = False

    while not stopits:
        
        ############
        # Run E-Step    
        suffstats = (e_step(documents, mu, sigma, lambd, beta))
        # Unpack results
        sigma_ss = suffstats.get('sigma')
        lambd = suffstats.get('lambd')
        beta_ss = suffstats.get('beta')
        bound_ss = suffstats.get('bound')
        print("Completed E-Step ({} seconds). \n".format(math.floor((time.process_time()-t1))))

        ############
        # Run M-Step 

        t1 = time.process_time()
        
        mu = opt_mu(
            lambd,
            covar=settings['covariates']['X'],
            enet=settings['gamma']['enet'],
            ic_k=settings['gamma']['ic.k'],
            maxits = settings['gamma']['maxits'],
            mode = settings['gamma']['mode']
        )

        sigma = opt_sigma(
            nu = sigma_ss,
            lambd = lambd, 
            mu = mu['mu'], 
            sigprior = settings['sigma']['prior']
        )
        
        beta = opt_beta(
            beta_ss, 
            kappa = None,
            #settings
        )
        print("Completed M-Step ({} seconds). \n".format(math.floor((time.process_time()-t1))))

        convergence = convergence_check(bound_ss, convergence, settings)
        stopits = convergence['stopits']

    ############
    #Step 3: Construct Output
    ############

    return {'lambd':lambd, 'beta_ss':beta, 'sigma_ss':sigma, 'bound_ss':bound}

def opt_mu(lambd, covar, enet, ic_k, maxits, mode = "L1"):
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
    
def opt_sigma(nu, lambd, mu, sigprior):
    #find the covariance
    # if ncol(mu) == 1: 
    #     covariance = np.cross(sweep(lambd, 2, STATS=as.numeric(mu), FUN="-")
    # else: 
    covariance = (lambd - mu).T@(lambd-mu)
    sigma = (covariance + nu)/lambd.shape[1]
    sigma = np.diag(np.diag(sigma))*sigprior + (1-sigprior)*sigma
    get_type(sigma)
    return sigma

def opt_beta(beta_ss, kappa):
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
    return beta

def convergence_check(bound_ss, convergence, settings):
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


""" Create Example """




""" Ingest data to create documents and vocab"""
# raw documents
data = pd.read_csv('data/poliblogs2008.csv')
# selection for quick testing
data = data[3000:4000]
# load preprocessed corpus
# 
documents = corpora.MmCorpus('data/corpus.mm')
# dictionary: 
dictionary = corpora.Dictionary.load('data/dictionary')
# vocabulary: 
vocab = dictionary.token2id

""" Setting control variables"""
prevalence = 'rating'
content = 'blog'
num_topics = 10
xmat = makeTopMatrix(prevalence, data)

yvar = makeTopMatrix(content, data).astype('category')
yvarlevels = set(yvar)
betaindex = yvar.cat.codes
A = len(set(betaindex))

interactions = True #settings.kappa
verbose = True

init_type = "Random" #settings.init
ngroups = 1 #settings.ngroups
max_em_its = 5 #settings.convergence
emtol = 1e-5 #settings.convergence

#gamma_prior=("Pooled","L1") # settings.gamma.prior
sigma_prior=0 #settings.sigma.prior
#kappa_prior=("L1","Jeffreys") # settings.kappa.prior

#Initialize parameters

settings = {
    'dim':{
        'K': num_topics, #number of topics
        'V' : len(dictionary), #number of words
        'A' : A, #dimension of topical content
        'N' : len(documents),
    },
    'verbose':verbose,
    'kappa':{
        'interactions':True,
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
        'X':xmat,
        'betaindex':betaindex,
        'yvarlevels':yvarlevels,
        'formula': prevalence,},
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



out = stm_control(documents, vocab, settings)
