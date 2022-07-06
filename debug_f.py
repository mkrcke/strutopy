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

#### DEBUG HESSIAN TO BE SYMMETRIC P.D.
# eta_test = np.array([-0.87537546, -0.12768776, -0.49046391,  0.08240985,  0.15822752,
#         0.207047  ,  0.13481039,  0.67539012,  0.01956233,  0.        ])
# beta_doc_kv_test = np.array([[4.91927715e-05, 2.82456027e-07, 1.76252870e-12, 7.49117325e-05,
#         4.32418559e-10, 1.58249267e-04, 2.10118064e-23, 1.33807769e-04,
#         1.35224920e-02, 1.87119687e-04, 8.98730431e-05, 4.14464224e-04,
#         6.52672919e-04, 7.11128177e-13, 3.78349568e-09, 4.22670805e-10,
#         3.08426553e-03, 1.25733398e-02, 7.65641796e-06, 5.56303437e-17,
#         2.35739130e-08, 6.48908989e-05, 3.57513155e-04, 1.08977662e-05,
#         5.75707816e-05, 3.26532543e-03, 3.95550772e-09, 1.34085915e-04,
#         3.97764166e-07, 5.37599216e-07, 4.83822807e-04, 9.69962218e-12,
#         6.02515716e-06, 6.96213136e-10, 3.72992236e-09, 2.76469557e-10,
#         2.28707513e-06, 5.22563373e-08, 1.86481685e-06, 7.50026210e-09,
#         6.83253693e-09, 2.30794200e-12, 2.28729640e-04, 1.72471893e-03],
#        [6.92288003e-03, 1.36218187e-03, 1.01547830e-05, 3.78168127e-07,
#         5.64955885e-04, 1.10096763e-06, 3.29138775e-18, 5.18603861e-03,
#         1.30549163e-02, 2.64020673e-07, 6.11227157e-10, 1.68215617e-03,
#         9.83194577e-04, 1.11054933e-07, 2.77545424e-03, 1.91492393e-08,
#         4.48084228e-03, 1.16447247e-02, 1.77197450e-15, 1.33736855e-10,
#         4.55143779e-13, 8.92594665e-13, 7.41545287e-10, 1.62212637e-02,
#         7.66670991e-07, 2.47468917e-03, 2.25314092e-14, 1.33968026e-12,
#         9.59288862e-04, 6.40025281e-11, 5.95530084e-05, 1.66195049e-08,
#         1.88777662e-03, 2.15848630e-03, 8.30528513e-04, 5.86978941e-06,
#         3.79201154e-10, 1.49332901e-04, 9.65977616e-13, 8.55870848e-04,
#         2.41643732e-05, 2.43199933e-11, 4.32389727e-08, 5.22347691e-07],
#        [8.53279704e-10, 1.35458184e-03, 2.70336338e-03, 2.01964590e-09,
#         4.81080141e-04, 3.99507411e-05, 3.83173625e-03, 1.72866027e-06,
#         1.85718602e-07, 3.27795641e-03, 4.10172480e-03, 3.23403013e-03,
#         5.40149994e-06, 4.01840497e-06, 5.79802242e-04, 1.01591852e-07,
#         3.36353757e-11, 9.29179866e-14, 2.93524526e-14, 2.97869785e-04,
#         2.67282961e-03, 5.14233198e-04, 2.06250228e-03, 4.04106064e-06,
#         1.12820932e-03, 7.18256076e-04, 2.32618870e-04, 3.01491805e-22,
#         9.59419788e-04, 1.15774563e-03, 5.86202993e-09, 3.87783294e-06,
#         1.31917120e-03, 1.31070772e-07, 2.76982903e-05, 2.92212730e-03,
#         1.88040852e-04, 6.52359085e-09, 4.10690672e-03, 7.36084701e-05,
#         2.76553098e-04, 1.02456110e-03, 2.32871933e-05, 9.55144383e-04],
#        [1.41770265e-03, 3.05009358e-06, 2.04638554e-18, 1.58172660e-02,
#         2.27128443e-04, 1.57374222e-03, 8.12502281e-09, 1.67290593e-15,
#         1.16598565e-03, 4.23585577e-08, 1.87639256e-05, 2.39281646e-14,
#         4.78751735e-06, 2.10463015e-10, 4.63066163e-03, 1.64123450e-06,
#         1.24467361e-03, 5.06948738e-13, 5.12014677e-04, 8.45527996e-11,
#         1.41661465e-06, 9.37954223e-08, 2.94198815e-04, 1.15393109e-03,
#         8.85202510e-07, 6.69437950e-03, 3.67997875e-02, 1.20057233e-04,
#         5.25218810e-02, 1.21762309e-04, 2.70157890e-04, 1.13542227e-06,
#         2.72205983e-13, 1.26213520e-08, 8.76903852e-05, 1.44326822e-04,
#         4.91825713e-04, 9.89380157e-05, 7.18088694e-05, 1.70790562e-04,
#         3.63284846e-08, 9.21344655e-13, 1.72342503e-07, 1.05435354e-03],
#        [6.85835860e-17, 9.54181722e-03, 1.12082311e-03, 5.08721842e-22,
#         1.34408522e-05, 1.54427836e-04, 1.72562053e-15, 5.46770965e-12,
#         8.88654635e-12, 1.32951186e-02, 2.97906285e-06, 1.46869915e-04,
#         5.06131203e-04, 3.07229325e-04, 1.13996462e-08, 4.60730414e-08,
#         2.19941022e-02, 4.23054531e-05, 1.52147862e-05, 9.24541428e-05,
#         4.28857130e-04, 9.89084038e-04, 2.10505580e-09, 1.43914469e-02,
#         3.57748523e-02, 2.00874398e-03, 3.63055201e-12, 3.16931348e-05,
#         3.80743341e-13, 2.92745689e-02, 4.10680825e-07, 3.45585687e-06,
#         2.08412536e-05, 6.45643557e-26, 1.75602877e-06, 7.68906911e-03,
#         5.34571881e-10, 3.94890469e-03, 3.36774929e-10, 1.58205757e-04,
#         8.79716482e-04, 3.71505509e-03, 4.79422645e-03, 1.14140930e-05],
#        [6.42810449e-09, 4.89767466e-17, 8.73065557e-15, 3.47186249e-06,
#         3.58507619e-13, 3.42454939e-05, 1.95679156e-05, 1.92556471e-06,
#         1.18090498e-07, 1.47349347e-03, 2.74190563e-12, 1.13001887e-05,
#         7.20783648e-06, 5.08495547e-06, 2.22588395e-09, 7.55645186e-05,
#         1.42242672e-10, 1.25490861e-04, 8.23540081e-03, 8.15131416e-03,
#         3.08377972e-04, 9.17959514e-07, 1.30077077e-04, 2.22723148e-02,
#         4.62678680e-04, 1.19665791e-11, 2.85741561e-16, 1.43197193e-02,
#         1.10123012e-05, 4.85186492e-10, 3.26260466e-04, 1.77260605e-03,
#         3.52749186e-03, 3.52989730e-09, 1.14352754e-06, 8.29482523e-04,
#         4.91903771e-06, 9.62257971e-02, 3.61095541e-02, 9.68584512e-10,
#         1.27297319e-08, 7.23218739e-09, 4.18251943e-11, 4.26489499e-07],
#        [1.12218101e-04, 8.20437723e-07, 2.91796577e-03, 9.04995702e-10,
#         9.64332888e-08, 4.50352146e-04, 2.53510958e-06, 4.84245126e-07,
#         9.91083175e-12, 1.75529372e-11, 4.79583962e-04, 9.79867097e-06,
#         1.31614266e-02, 7.76669495e-03, 6.90298835e-05, 8.48966294e-05,
#         5.96722246e-05, 4.20358875e-03, 1.85535588e-04, 1.65610104e-03,
#         2.46917663e-15, 9.84253973e-03, 1.81662535e-06, 1.98681811e-07,
#         1.33827309e-09, 3.74751823e-03, 2.25285219e-03, 1.41218311e-05,
#         6.79981915e-08, 1.30843053e-21, 3.64488433e-04, 8.62873371e-05,
#         3.73743753e-04, 3.22183082e-04, 3.80181487e-03, 1.97739104e-03,
#         1.33973051e-03, 1.77784530e-09, 2.50429328e-07, 6.81332209e-10,
#         2.01484935e-04, 6.96810585e-06, 4.85488892e-04, 1.15927055e-04],
#        [1.34978173e-04, 4.82814767e-11, 8.55197217e-09, 9.88487209e-09,
#         2.56905014e-05, 3.82874618e-03, 5.96020898e-03, 1.15534674e-02,
#         8.21017276e-12, 1.40899772e-03, 7.93775297e-03, 1.17308108e-08,
#         3.27713590e-02, 1.70289630e-13, 2.98437522e-03, 8.70788912e-05,
#         1.30076319e-03, 9.41471224e-03, 3.55197761e-04, 2.81021829e-19,
#         1.22331560e-02, 1.93326810e-10, 1.95974151e-04, 2.21360069e-04,
#         3.20630522e-05, 1.62672210e-12, 2.95254620e-11, 5.53978452e-03,
#         6.28764897e-03, 7.92956484e-10, 7.69479459e-06, 3.77631703e-03,
#         1.25732376e-09, 2.10894732e-03, 8.03899302e-07, 1.12029867e-02,
#         1.48153455e-02, 2.94782082e-08, 1.34834725e-05, 7.27261347e-10,
#         8.84892159e-06, 3.16632931e-03, 6.08566194e-02, 1.14377199e-03],
#        [6.46142714e-07, 3.57095266e-08, 1.89249762e-04, 5.23736884e-04,
#         1.08045329e-07, 1.06353370e-10, 1.02510470e-03, 1.42981260e-14,
#         1.76857340e-08, 1.17247657e-08, 9.07562579e-09, 1.25158779e-02,
#         3.77681299e-05, 1.73192889e-04, 2.87979185e-15, 4.94785737e-04,
#         2.07805952e-07, 5.00813719e-09, 7.56672126e-07, 7.01233738e-11,
#         5.68605363e-09, 3.57534146e-10, 1.02668105e-02, 6.42892855e-03,
#         2.13594495e-04, 5.06615250e-05, 7.08635901e-07, 2.88934966e-06,
#         5.95086298e-06, 4.02855965e-03, 4.88497151e-09, 1.11886070e-06,
#         3.75956897e-07, 2.08593405e-03, 6.56499423e-04, 4.64067365e-06,
#         1.87678220e-05, 7.38598182e-03, 3.80309689e-03, 3.95453329e-07,
#         1.85004529e-02, 6.96821597e-14, 6.88388082e-03, 6.85065249e-06],
#        [1.61042254e-10, 7.85174070e-10, 2.53218983e-03, 8.17961442e-11,
#         1.01387156e-02, 9.16396882e-05, 2.17044954e-05, 5.13392504e-08,
#         2.15282568e-14, 4.43835678e-21, 2.90992364e-04, 5.85893888e-08,
#         4.35437265e-07, 7.28999545e-13, 1.88387525e-08, 9.66895167e-19,
#         2.34224914e-14, 2.56809344e-09, 1.89023123e-02, 1.68381769e-07,
#         7.44995472e-08, 1.84419505e-05, 4.36984842e-06, 1.94381418e-03,
#         5.11992588e-03, 3.55581466e-04, 4.01921489e-02, 3.17313037e-03,
#         5.43763206e-12, 1.27799540e-04, 6.27741716e-08, 1.65799748e-10,
#         1.11319401e-04, 4.00136853e-05, 3.94909147e-07, 3.76462293e-12,
#         1.43714303e-06, 1.23369456e-07, 7.78870214e-10, 8.50194670e-04,
#         1.58796964e-14, 1.67046456e-02, 2.66807016e-04, 4.95836300e-14]])
# word_count_test = np.array([1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2,
#        1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1])

# def hessian_test(eta, word_count, beta_doc_kv):
#     eta_ = np.insert(eta, K - 1, 0)
#     theta = stable_softmax(eta_)
    
#     a = np.transpose(np.multiply(np.transpose(beta_doc_kv), np.exp(eta_)))  # KxV
#     b = np.multiply(a, np.transpose(np.sqrt(word_count))) / np.sum(a, 0)  # KxV
#     c = np.multiply(b, np.transpose(np.sqrt(word_count)))  # KxV

#     hess = b @ b.T - np.sum(word_count) * np.multiply(
#         theta[:,None], theta[None,:]
#     )
#     assert check_symmetric(hess), 'hessian is not symmetric'
#     # broadcasting, works fine
#     # difference to the c++ implementation comes from unspecified evaluation order: (+) instead of (-)
#     np.fill_diagonal(
#         hess, np.diag(hess) - np.sum(c, axis=1) + np.sum(word_count)*theta
#     )

#     d = hess[:-1, :-1]
#     f = d + siginv   
#     return f
    
# hessian_test(eta=eta_test[:-1], word_count=word_count_test, beta_doc_kv=beta_doc_kv_test)

# eta = eta_test[:-1]
# eta = np.insert(eta,K-1, 0)
# theta = stable_softmax(eta)
# a = np.transpose(np.multiply(beta_doc_kv_test.T, np.exp(eta)))
# a.shape #KxV
# b = (np.multiply(a, np.sqrt(word_count_test)) / np.sum(a, 0))
# b.shape #KxV
# c = np.multiply(b, np.transpose(np.sqrt(word_count_test)))  # KxV
# c.shape #KxV
# hess = b @ b.T - np.sum(word_count_test) * np.dot(theta[:,None], theta[None,:])
# ## arma::mat hess = EB * EB.t() - sum(doc_cts) * (theta * theta.t()); /

# # diagonal entries: 
# np.fill_diagonal(
#         hess, np.diag(hess) - np.sum(c, axis=1) + np.sum(word_count_test)*theta
#     )
# hess = hess[:-1,:-1]+siginv
