import numpy as np
import numpy.random as random
import scipy
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt

# three dimension plot
mu_test = np.array([0,0])
K = 3
V = 50
doc_ct = np.ones(V)
beta_init = random.gamma(0.1, 1, V*K).reshape(K,V)
beta_test = beta_init / np.sum(beta_init, axis=1)[:, None]
sigma_test = np.zeros(((K - 1), (K - 1)))
np.fill_diagonal(sigma_test, 20)
sigobj_test = np.linalg.cholesky(sigma_test)  # initialization of sigma not positive definite
siginv_test = np.linalg.inv(sigobj_test).T * np.linalg.inv(sigobj_test)

def f_test(eta, word_count, beta_doc_kv, siginv=siginv_test, mu = mu_test):
    # precomputation
    eta = np.insert(eta, K - 1, 0)
    Ndoc = int(np.sum(word_count))
    # formula
    # from cpp implementation:
    # log(expeta * betas) * doc_cts - ndoc * log(sum(expeta))
    part1 = np.dot(
        word_count, (eta.max() + np.log(np.exp(eta - eta.max()) @ beta_doc_kv))
    ) - Ndoc * scipy.special.logsumexp(eta)
    part2 = 0.5 * (eta[:-1] - mu).T @ siginv @ (eta[:-1] - mu)
    return np.float32(part2 - part1)

x=np.linspace(-2,2,200)
y=np.linspace(-2,2,200)
X, Y = np.meshgrid(x, y)

Z = []
range_ = range(200)
for i in range_:
    eta_i = np.array([X[i],Y[i]])
    Z.append([f_test(eta = eta_i.T[val], word_count=doc_ct, beta_doc_kv=beta_test) for val in range_])

# find minimum
idy, idx = np.unravel_index(np.argmin(np.array(Z)), np.array(Z).shape)
print(f'found a minimum of {np.min(np.array(Z))} for x = {x[idx]} and y = {y[idy]}')

# does the optimization find the minimum? 
def optimize_eta(eta, word_count, beta_doc_kv, mu=mu_test, siginv=siginv_test):
    def f_test(eta, word_count, beta_doc_kv):
        # precomputation
        eta = np.insert(eta, K - 1, 0)
        Ndoc = int(np.sum(word_count))
        # formula
        # from cpp implementation:
        # log(expeta * betas) * doc_cts - ndoc * log(sum(expeta))
        part1 = np.dot(
            word_count, (eta.max() + np.log(np.exp(eta - eta.max()) @ beta_doc_kv))
        ) - Ndoc * scipy.special.logsumexp(eta)
        part2 = 0.5 * (eta[:-1] - mu).T @ siginv @ (eta[:-1] - mu)
        return np.float32(part2 - part1)

    def df_test(eta, word_count, beta_doc_kv):
        """gradient for the objective of the variational update q(etas)"""
        # precomputation
        eta = np.insert(eta, K - 1, 0)
        # formula
        # part1 = np.delete(np.sum(phi * word_count,axis=1) - Ndoc*theta, K-1)
        # part1 = np.delete(np.sum(phi * word_count,axis=1) - Ndoc*theta, K-1)
        part1 = np.delete(
            beta_doc_kv @ (word_count / np.sum(beta_doc_kv.T, axis=1))
            - np.sum(word_count) / np.sum(np.exp(eta)),
            K - 1,
        )
        part2 = siginv @ (eta[:-1] - mu)
        # We want to maximize f, but numpy only implements minimize, so we
        # minimize -f
        return np.float64(part2 - part1)

    return optimize.minimize(
        f_test, x0=eta, args=(word_count, beta_doc_kv), jac=df_test, method="BFGS", options={'maxiter': 500, 'gtol': 1e-01, 'eps':1},
    ).x

result = optimize_eta(eta=np.zeros(2), word_count=doc_ct, beta_doc_kv=beta_test, mu = mu_test)
print(f'optimal eta using numerical optimizationfound: {result}')


# show plot
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 200, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('objective surface')
plt.savefig('obj.png', dpi=400)
plt.show()