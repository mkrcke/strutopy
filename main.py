import pandas as pd
import numpy as np
from gensim import corpora
import time
import math

from stm import STM

###main.py
""" Create Example """
""" Ingest data to create documents and vocab"""
# raw documents
data = pd.read_csv('data/poliblogs2008.csv')
# selection for quick testing

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



#Initialize parameters
data = data[3000:4000]
xmat = data.loc[:, prevalence]
yvar = data.loc[:, content].astype('category')
yvarlevels = set(yvar)
betaindex = np.array(yvar.cat.codes)
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
#Random initialization, TO-DO: Improve initialization 

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


def stm_control(documents, settings, model=None):
    ##########
    #Step 1: Initialize Parameters
    model = STM(settings, documents, dictionary)
    ##########

    # # unpack initialized model
    # mu = model['mu']
    # sigma = model['sigma']
    # lambd = model['lambda'] 
    # beta = {'beta': model['beta'],
    #         'kappa': model['kappa']}
    # convergence = None
    # #discard the old object
    # del model

    ############
    #Step 2: Run EM
    ############
    t1 = time.process_time()
    suffstats = [] 
    stopits = False

    while not stopits:
        
        ############
        # Run E-Step    
        sigma_ss, lambd, beta_ss, bound_ss = model.e_step(documents)
        # Unpack results
        
        print("Completed E-Step ({} seconds). \n".format(math.floor((time.process_time()-t1))))

        ############
        # Run M-Step 

        t1 = time.process_time()
        
        mu = model.opt_mu(
            lambd,
            covar=settings['covariates']['X'],
            enet=settings['gamma']['enet'],
            ic_k=settings['gamma']['ic.k'],
            maxits = settings['gamma']['maxits'],
            mode = settings['gamma']['mode']
        )

        sigma = model.opt_sigma(
            nu = sigma_ss,
            lambd = lambd, 
            mu = mu['mu'], 
            sigprior = settings['sigma']['prior']
        )
        
        beta = model.opt_beta(
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

    return {'lambd':lambd, 'beta_ss':beta, 'sigma_ss':sigma, 'bound_ss':bound_ss}

stm_control(documents, settings, model=None)