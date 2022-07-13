#%%
import json
import pandas as pd
import numpy as np
from stm import STM 
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import (remove_stopwords,
                                          preprocess_string,
                                          strip_numeric,
                                          stem_text,
                                          strip_punctuation)
from gensim.utils import simple_preprocess
from gensim import corpora
from operator import itemgetter


def label_topics(model, n, topics):
    """
    Label topics
    
    Generate a set of words describing each topic from a fitted STM object.
    
    Highest Prob: are the words within each topic with the highest probability
    (inferred directly from topic-word distribution parameter beta)
    
    @param model An \code{STM} model object.
    @param topics vector of numbers indicating the topics to include.  Default
    is all topics.
    @param n The desired number of words (per type) used to label each topic.
    Must be 1 or greater.

    @return labelTopics object (list) \item{prob }{matrix of highest
    probability words}
    """
    beta = model.beta
    copy = beta.copy()
    K = model.K
    vocab = dictionary

    # wordcounts = model.settings["dim"]["wcounts"]["x"] #TODO: implement word counts
    
    # Sort by word probabilities on each row of beta
    # Returns words with highest probability per topic
    problabels = np.argsort(-1*beta)[:10]

    out = []
    for k in range(K):
        probwords = [itemgetter(i)(vocab) for i in problabels[k,:n]]
        print(f"Topic {k}:\n \t Highest Prob: {probwords}")
        out.append(probwords)
    
    return

#%% fit model
# load raw documents
data = pd.read_csv('data/poliblogs2008.csv')
data = data[3000:3500]
# %% Preprocess text
data.documents = data.documents.apply(remove_stopwords)
data.documents = data.documents.apply(strip_numeric)
data.documents = data.documents.apply(strip_punctuation)
data.documents = data.documents.apply(stem_text)
# %% create dictionary and corpus
doc_tokens = [simple_preprocess(doc) for doc in data.documents]
dictionary = corpora.Dictionary(doc_tokens)
documents = [dictionary.doc2bow(doc) for doc in doc_tokens]
# %% fit model 
num_topics = 10 
verbose = True

# set prevalence and content variable
prevalence = 'rating'
content = 'blog'
xmat = data.loc[:, prevalence]
yvar = data.loc[:, content].astype('category')
yvarlevels = set(yvar)
betaindex = np.array(yvar.cat.codes)
A = len(set(betaindex))

settings = {
    'dim':{
        'K': 10, #number of topics
        'V' : len(dictionary), #number of words
        'A' : A, #dimension of topical content
        'N' : len(documents),
    },
    'verbose': verbose,
    'kappa':{
        'interactions':False,
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
        'mode':'spectral', 
        'nits':20,
        'burnin':25,
        'alpha':50/num_topics,
        'eta':.01,
        's':.05,
        'p':3000},
    'convergence':{
        'max.em.its':15,
        'em.converge.thresh':1e-5,
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
        'prior':0,
        'ngroups':1,},
}

#%% run model 
model = STM(settings, documents, dictionary)
model.expectation_maximization(saving=False)

# %%
label_topics(model, n=5, topics=3)
