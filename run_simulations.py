import numpy as np
import matplotlib.pyplot as plt
import time
import math
import json

# custom packages

from stm import STM
from simulate import generate_docs

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

def stm_control(documents, settings, model=None):
    np.random.seed(123)
    ##########
    #Step 1: Initialize Parameters
    if model is None:
        model = STM(settings, documents, dictionary)

    ############
    #Step 2: Run EM
    ############
    global_time = time.process_time()
    stopits = False
    convergence = None
    while not stopits:
        t1 = time.process_time()
        ############
        # Run E-Step    
        sigma_ss, beta_ss, bound_ss = model.e_step(documents)
        print("Completed E-Step ({} seconds). \n".format(math.floor((time.process_time()-t1))))

        ############
        # Run M-Step 

        t1 = time.process_time()
        
        mu = model.update_mu(
            covar=settings['covariates']['X'],
            enet=settings['gamma']['enet'],
            ic_k=settings['gamma']['ic.k'],
            maxits = settings['gamma']['maxits'],
            mode = settings['gamma']['mode']
        )

        sigma = model.update_sigma(
            nu = sigma_ss, 
            mu = mu, 
            sigprior = settings['sigma']['prior']
        )
        
        beta = model.update_beta(
            beta_ss, 
            kappa = None,
            #settings
        )
        print("Completed M-Step ({} seconds). \n".format(math.floor((time.process_time()-t1))))

        convergence = model.convergence_check(bound_ss, convergence, settings)
        stopits = convergence['stopits']

    ############
    #Step 3: Construct Output
    ############
    time_processed = time.process_time()-global_time
    model = {'mu':mu, 'sigma':sigma, 'beta':beta, 'convergence':convergence, 'settings': settings, 'time_processed':time_processed}
    json.dump(model, open("stm_model.json", "w"), cls=NumpyEncoder)

    return 

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
documents, vocabulary = basic_simulations(n_docs=100, n_words=100, V=500, ATE=.2, alpha=np.array([.3,.4,.3]), display=False)
betaindex = np.concatenate([np.repeat(0,50), np.repeat(1,50)])
num_topics = 3
dictionary=np.arange(vocabulary)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
        
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

stm_control(documents, settings, model=None)




