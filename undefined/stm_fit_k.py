
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
from heldout import find_k 
import numpy as np
import pandas as pd 

# load corpus & dictionary
data = pd.read_csv('application/data/corpus_preproc.csv')
corpus = corpora.MmCorpus('application/data/BoW_corpus.mm')
dictionary = corpora.Dictionary.load('application/data/dictionary.mm')

# set topical prevalence 
prevalence = ['statistics','ml']
xmat = data.loc[:, prevalence]
interactions = False #settings.kappa
verbose = True

init_type = "spectral" #settings.init
ngroups = 1 #settings.ngroups
max_em_its = 5 #settings.convergence
emtol = 1e-5 #settings.convergence
sigma_prior=0 #settings.sigma.prior

settings = {
    'dim':{
        'K': None, #number of topics
        'V' : len(dictionary), #number of words
        #'A' : A, #dimension of topical content
        'N' : len(corpus),
    },
    'verbose':verbose,
    'kappa':{
        'interactions':False,
        'fixedintercept': True,
        'contrats': False,
        'mstep': {'tol':0.01, 'maxit':5}},
    'convergence':{
        'max.em.its':max_em_its,
        'em.converge.thresh':emtol,
        'allow.neg.change':True,},
    'covariates':{
        'X':xmat,}, # this is were the topical prevalence covariates are included 
    'gamma':{
        'prior':np.nan, #sigma in the topical prevalence model
        'enet':1, #regularization term
        'ic.k':2,#information criterion
        'maxits':1000,},
    'sigma':{
        'prior':0,
        'ngroups':1,},
}


def main():
    # define k candidates
    K_candidates = np.array([10,20,30,40,50,60,70,80,90,100])

    results_k = find_k(K_candidates, corpus=corpus, settings=settings,models=['STM'])
    
    # plot results
    fig, ax = plt.subplots()

    ax.scatter(K_candidates, results_k[0], label='STM')

    plt.title("Held-out Likelihood for varying number of topics")
    plt.xlabel("Number of topics")
    plt.ylabel("Held-out Likelihood")
    plt.legend()
    #plt.savefig('img/different_k_no_treatment', bbox_inches='tight', dpi=360)
    plt.show()
    return 

    
if __name__ == "__main__":
    main()