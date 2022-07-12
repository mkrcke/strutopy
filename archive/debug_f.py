import numpy as np
import numpy.random as random
import scipy
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
### define input
import csv


# desired input: eta, fn, gr, doc_ct, mu=mu, siginv=siginv, beta_doc
K = 30
V = 143
word_count = np.ones(V)
eta = np.zeros(K-1)
mu = np.zeros(K-1)
beta_doc_kv = pd.read_csv('np.txt', sep=" ", header=None).values
sigma = np.zeros(((K-1), (K-1)))
np.fill_diagonal(sigma, 20)
sigobj = np.linalg.cholesky(sigma)  # initialization of sigma not positive definite
siginv = np.linalg.inv(sigobj).T * np.linalg.inv(sigobj)
sigmaentropy = np.sum(np.log(np.diag(sigobj)))

def f(eta, word_count, beta_doc_kv):
    # precomputation
    eta = np.insert(eta, K - 1, 0)
    Ndoc = int(np.sum(word_count))
    # formula
    # from cpp implementation:
    # log(expeta * betas) * doc_cts - ndoc * log(sum(expeta))
    return np.float64((0.5 * (eta[:-1] - mu).T @ siginv @ (eta[:-1] - mu)) - (np.dot(
        word_count, eta.max() + np.log(np.exp(eta - eta.max()) @ beta_doc_kv))
     - Ndoc * scipy.special.logsumexp(eta)))

def df(eta, word_count, beta_doc_kv):
    """gradient for the objective of the variational update q(etas)"""
    # precomputation
    eta = np.insert(eta, K - 1, 0)
    # formula
    # part1 = np.delete(np.sum(phi * word_count,axis=1) - Ndoc*theta, K-1)
    # part1 = np.delete(np.sum(phi * word_count,axis=1) - Ndoc*theta, K-1)
    return np.array(np.float64(siginv @ (eta[:-1] - mu)-(beta_doc_kv @ (word_count / np.sum(beta_doc_kv.T, axis=1))
        - (np.sum(word_count) / np.sum(np.exp(eta)))*np.exp(eta))[:-1]))
    # We want to maximize f, but numpy only implements minimize, so we
    # minimize -f
    # print(part1)
    # print(part2)
    # return np.float64(part2 - part1)

### requirements
def stable_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    xshift = x - np.max(x)
    exps = np.exp(xshift)
    return exps / np.sum(exps)


def softmax_weights(x, weight):
    """Compute weighted softmax values for each sets of scores in x."""
    xshift = x - np.max(x)
    exps = weight * np.exp(xshift)[:, None]
    return exps / np.sum(exps)


def optimize_eta(eta, word_count, beta_doc_kv):
    def f(eta,  word_count, beta_doc_kv):
        # precomputation
        eta = np.insert(eta, K - 1, 0)
        Ndoc = int(np.sum(word_count))
        part1 = np.dot(
            word_count, (eta.max() + np.log(np.exp(eta - eta.max()) @ beta_doc_kv))
        ) - Ndoc * scipy.special.logsumexp(eta)
        part2 = 0.5 * (eta[:-1] - mu).T @ siginv @ (eta[:-1] - mu)
        print(part2 - part1)
        return np.float32(part2 - part1)
    def df(eta,  word_count, beta_doc_kv):
        """gradient for the objective of the variational update q(etas)"""
        # precomputation
        eta = np.insert(eta, K - 1, 0)
        # formula
        part1 = np.delete(
            beta_doc_kv @ (word_count / np.sum(beta_doc_kv.T, axis=1))
            - np.sum(word_count) / np.sum(np.exp(eta)),
            K - 1,
        )
        part2 = siginv @ (eta[:-1] - mu)
        # We want to maximize f, but numpy only implements minimize, so we
        # minimize -f
        print(part2 - part1)
        return (part2-part1)

    return optimize.minimize(
        f, x0=eta, args=(word_count, beta_doc_kv), jac=df, options={'maxiter': 500, 'gtol': 1e-5, 'eps':10},
    )


def make_pd(M):
    dvec = M.diagonal()
    magnitudes = np.sum(abs(M), axis=0) - abs(dvec)
    # cholesky decomposition works only for symmetric and positive definite matrices
    dvec = np.where(dvec < magnitudes, magnitudes, dvec)
    # A Hermitian diagonally dominant matrix A with real non-negative diagonal entries is positive semidefinite.
    np.fill_diagonal(M, dvec)
    return M


def hessian(eta):
    eta_ = np.insert(eta, K - 1, 0)
    theta = stable_softmax(eta_)
    
    a = np.transpose(np.multiply(np.transpose(beta_doc_kv), np.exp(eta_)))  # KxV
    b = np.multiply(a, np.transpose(np.sqrt(word_count))) / np.sum(a, 0)  # KxV
    c = np.multiply(b, np.transpose(np.sqrt(word_count)))  # KxV

    hess = b @ b.T - np.sum(word_count) * np.multiply(
        theta[:,None], theta[None,:]
    )
    assert check_symmetric(hess), 'hessian is not symmetric'
    # broadcasting, works fine
    # difference to the c++ implementation comes from unspecified evaluation order: (+) instead of (-)
    np.fill_diagonal(
        hess, np.diag(hess) - np.sum(c, axis=1) + np.sum(word_count)*theta
    )

    d = hess[:-1, :-1]
    f = d + siginv   
    return f


def decompose_hessian(hess):
    try:
        L = np.linalg.cholesky(hess)
    except:
        try:
            L = np.linalg.cholesky(make_pd(hess))
            print("converts Hessian via diagonal-dominance")
        except:
            L = np.linalg.cholesky(make_pd(hess) + 1e-5 * np.eye(hess.shape[0]))
            print("adds a small number to the hessian")
    return L


def optimize_nu(L):
    nu = np.linalg.inv(
        np.triu(L.T)
    )  # watch out: L is already a lower triangular matrix!
    nu = nu @ nu.T
    return nu


def lower_bound(L, eta_):
    eta_ = np.insert(eta, K - 1, 0)
    theta = stable_softmax(eta_)
    # compute 1/2 the determinant from the cholesky decomposition
    detTerm = -np.sum(np.log(L.diagonal()))
    diff = eta - mu
    ############## generate the bound and make it a scalar ##################
    beta_temp_kv = beta_doc_kv * np.exp(eta_)[:, None]
    bound = (
        np.log(
            theta[
                None:,
            ]
            @ beta_temp_kv
        )
        @ word_count
        + detTerm
        - 0.5 * diff.T @ siginv @ diff
        - sigmaentropy
    )
    return bound


def update_z(eta, beta_doc_kv, word_count):
    """Compute the update for the variational latent parameter z

    Args:
        eta (np.array): 1D-array representing prior to the document-topic distribution
        beta_doc_kv (np.array): 2D-array (K by V) containing the topic-word distribution for a specific document

    Returns:
        phi: update for the variational latent parameter z
    """
    eta_ = np.insert(eta, K - 1, 0)
    a = np.multiply(beta_doc_kv.T, np.exp(eta_)).T  # KxV
    b = np.multiply(a, (np.sqrt(word_count) / np.sum(a, 0)))  # KxV
    phi = np.multiply(b, np.sqrt(word_count).T)  # KxV
    return phi

def check_symmetric(M, rtol=1e-05, atol=1e-08):
    return np.allclose(M, np.transpose(M), rtol=rtol, atol=atol)

# compute values
f(eta, word_count, beta_doc_kv)  # fixed
df(eta, word_count, beta_doc_kv)  # fixed
hess = hessian(eta)  # fixedhess
L = decompose_hessian(hess)  # fixed
lower_bound(L, eta)  # fixed
optimize_nu(L)  # fixed
update_z(eta, beta_doc_kv, word_count)  # fixed


#test optimize
def print_fun(x):
    print("Current value: {}".format(x))

result = optimize.minimize(fun = f, x0=eta, args=(word_count, beta_doc_kv), jac=df, method="BFGS", options={'disp':True})
result.hess_inv
## invert via cholesky decomp
L = np.linalg.inv(np.linalg.cholesky(hess))
hess_inv = np.dot(L,np.transpose(L))
