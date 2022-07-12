import numpy as np 
import scipy
from scipy import optimize
import numpy.random as random
import matplotlib.pyplot as plt

### define input 
# desired input: eta, fn, gr, doc_ct, mu=mu, siginv=siginv, beta_doc
K = 30 
V = 143
word_count = np.ones(V)
eta = np.zeros(K-1)
mu = np.zeros(K-1)
beta_init = random.gamma(.1,1, V*K).reshape(K,V)
beta_doc_kv = (beta_init / np.sum(beta_init, axis=1)[:,None])
sigma = np.zeros(((K-1),(K-1)))
np.fill_diagonal(sigma, 20)
sigobj = np.linalg.cholesky(sigma) #initialization of sigma not positive definite
siginv = np.linalg.inv(sigobj).T*np.linalg.inv(sigobj)

### requirements 
#  
def stable_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    xshift = x-np.max(x)
    exps = np.exp(xshift)
    return exps / np.sum(exps)

def softmax_weights(x, weight):
    """Compute weighted softmax values for each sets of scores in x.""" 
    xshift = x - np.max(x)
    exps = weight*np.exp(xshift)[:,None]
    return exps / np.sum(exps)

### (re-)define function
def optimize_eta(eta,mu,word_count,beta_doc): 
        """Optimizes the variational parameter eta given the likelihood and the gradient function
        """
        def f(eta, word_count, beta_doc):
            """objective for the variational update q(eta)
            We want to maximize f, but numpy only implements minimize, so we minimize -f
            """
            # precomputation
            eta_ = np.insert(eta, K-1, 0)
            Ndoc = int(np.sum(word_count))
            #formula 
            # from cpp implementation: 
            # log(expeta * betas) * doc_cts - ndoc * log(sum(expeta))
            part1 = np.dot(word_count,(eta_.max() + np.log(np.exp(eta_ - eta_.max())@beta_doc)))-Ndoc*scipy.special.logsumexp(eta_)
            part2 = .5*(eta_[:-1]-mu).T@siginv@(eta_[:-1]-mu)
            return np.float32(part2 - part1)

        def df(eta, word_count, beta_doc):
            """gradient for the objective of the variational update q(etas)
            """
            # precomputation
            eta_ = np.insert(eta, K-1, 0)
            Ndoc = int(np.sum(word_count))
            theta = stable_softmax(eta_)
            phi = softmax_weights(eta_, beta_doc)
            #formula
            # part1 = np.delete(np.sum(phi * word_count,axis=1) - Ndoc*theta, K-1)
            part1 = np.delete(np.sum(phi * word_count,axis=1) - Ndoc*theta, K-1)
            part2 = siginv@(eta_[:-1]-mu)
            # We want to maximize f, but numpy only implements minimize, so we
            # minimize -f
            return np.float64(part2 - part1)
        return optimize.minimize(
            f, 
            x0=eta,
            args=(word_count, beta_doc),
            jac=df, 
            method="BFGS",
            options={'disp': True}
            )


### computation by hand

# current optimization results
eta_python = np.array([-94.66666552, -94.66666552, -94.66666552, -94.66666552,
       -94.66666552, -94.66666552, -94.66666552, -94.66666552,
       -94.66666552, -94.66666552, -94.66666552, -94.66666552,
       -94.66666552, -94.66666552, -94.66666552, -94.66666552,
       -94.66666552, -94.66666552, -94.66666552, -94.66666552,
       -94.66666552, -94.66666552, -94.66666552, -94.66666552,
       -94.66666552, -94.66666552, -94.66666552, -94.66666552,
       -94.66666552])
eta_test_py = np.insert(eta_python, K-1, 0)

# R optimization results
eta_R = np.array([-0.1913102, 0.4869240, 0.3968792, -0.1521476, -0.1631895, -0.1018726,  0.3824809,  0.2582114, -1.2964760,
-0.4088999, 0.7799313, -0.2220894, -0.1803936, -0.1649422,  0.2787281,  0.7493393, -0.2635081,  0.0765101,
0.2939820, -1.0320776,-0.7096768, -0.1484485, 0.2878102,-0.2820451, -0.3177538 , 0.2290855 , 1.0529572,0.4490692, 0.3583511])
eta_test_R = np.insert(eta_R, K-1,0)

### run function for zero initialisation
optimize_eta(
    eta=eta,
    mu=mu,
    word_count=word_count,
    beta_doc=beta_doc_kv
    )
### run function for python values
optimize_eta(
    eta=eta_test_py,
    mu=mu,
    word_count=word_count,
    beta_doc=beta_doc_kv
    ).fun
### run function for R values
optimize_eta(
    eta=eta_test_R[:-1],
    mu=mu,
    word_count=word_count,
    beta_doc=beta_doc_kv
    )


# debugging the gradient
theta = stable_softmax(eta)
phi = softmax_weights(eta, beta_doc_kv)
Ndoc = np.sum(word_count)

np.sum(phi * word_count,axis=1) - Ndoc*theta

# C++ yields the exact same result
# betas * (doc_cts / arma::trans(sum(betas, 0)))
# - (sum(doc_cts) / sum(expeta)) * expeta
beta_doc_kv @ (word_count / np.sum(beta_doc_kv).T)- (np.sum(word_count) / np.sum(np.exp(eta))) * np.exp(eta)


optimize_eta(
    eta=random.gamma(.1,1,K-1),
    mu=mu,
    word_count=word_count,
    beta_doc=beta_doc_kv
    )


## investigate functional form
values = random.gamma(.1,1,3000000).reshape(100000,30)
res = []

for x in values: 
    res.append(np.dot(word_count,(x.max() + np.log(np.exp(x - x.max())@beta_doc_kv)))-Ndoc*scipy.special.logsumexp(x))

plt.plot(np.mean(values, axis=1), np.sort(res))
plt.show()


