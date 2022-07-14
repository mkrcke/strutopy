#%%
import numpy as np
import numpy.random as random

# %%
def stable_softmax(x):
        """Compute softmax values for each sets of scores in x."""
        xshift = x - np.max(x)
        exps = np.exp(xshift)
        return exps / np.sum(exps)

def init_gamma(p, topics, mean=None):
    if mean==None: 
        mean = np.random.standard_normal(p)
    sigma_prior = np.eye(p)
    mean = np.random.multivariate_normal(mean, sigma_prior)
    sigma = np.eye(p)
    return np.random.multivariate_normal(mean, sigma, topics-1)

def metadata(n_docs, levels=[0,1],p=None):
    # simulate one-hot encoding x1==True iff x2==False
    x1 = random.choice(levels, size=int(n_docs), replace=True, p=None)
    x2 = np.array([int(i==False) for i in x1])
    x = np.column_stack((x1, x2))
    return x

def init_eta(x, gamma):
    mu = x@gamma.T
    sigma = np.eye(n_topics-1)
    eta = []
    for d in range(n_docs):
        eta.append(np.random.multivariate_normal(mu[d], sigma))
    eta = np.array(eta)
    return eta

def map_eta(eta): 
    eta_ = np.array(list(map(lambda x: np.insert(x,len(x),0), eta)))
    theta = np.array(list(map(lambda x: stable_softmax(x), eta_)))
    return theta

def doc_topic_dist(p, n_topics, n_docs):
    """Generate documents for varying hyperprior on topical prevalence

    Example: 
    if p = 2 and k=5, we need to draw a sample of 5 for each p
    result: how probable is each topic given p=1 and p=2 respectively.
    result: dimension of theta_k 5 x 1 (dim(theta) = 5 x 2)
    result: dimension of eta_k 4 x 1 (dim(eta) = 4 x 2)

    # instead of fixing the mean parameter for each topic, we choose a standard normal distribution
    # as of now, the covariance is fixed to zero and the variance is constant 1 over all topics

    Args:
        p (_type_): _description_
        n_topics (_type_): _description_
        n_docs (_type_): _description_
    """
    #simulate metadata
    x = metadata(n_docs, levels=[0,1])
    #simulate hyperprior
    gamma = init_gamma(p, n_topics)
    #create topic proportion
    eta = init_eta(x, gamma)
    theta = map_eta(eta)
    return theta

def word_topic_dist(beta, n_words):
    """2D numpy array containing the word-topic distribution

    Args:
        beta (nd.array): 2D-array of dimension K by V
    """
    if beta == None:
        beta = random.dirichlet(size=n_topics, alpha=np.repeat(0.05, n_words))
    else:
        beta = np.array(beta)            
        assert type(beta) == np.ndarray, 'beta needs to be a 2D numpy array'
    return beta

def sample_documents(n_docs):
    p = theta @ beta
    documents = []
    for doc in range(n_docs):
        doc_words = random.multinomial(n_words, p[doc], size=1)
        documents.append(
            list(
                zip(
                    np.asarray(doc_words).nonzero()[1],
                    doc_words[np.where(doc_words > 0)],
                )
            )
        )
    return documents

p = 2 #(two levels for the topical prevalence parameter) 
n_topics = 10 
n_docs = 500
n_words = 500
theta = doc_topic_dist(p,n_topics, n_docs)
beta = word_topic_dist(n_words = n_words, beta=None)
documents = sample_documents(n_docs)