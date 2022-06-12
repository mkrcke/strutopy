# This script contains the basic simulations for the STM package. 

import numpy as np
import matplotlib.pyplot as plt

n_topics = 3
n_doc = 100 
n_words = 40 # To-Do: sampling number of words per document (Poisson distribution)
ATE = .2
t_1 = (-ATE,0,+ATE)

# define concentration parameters
alpha_0 = np.array([0.3,0.4,0.3])
alpha_1 = np.add(alpha_0,t_1)
concentration_parameter = np.repeat(0.05, n_topics)
# sample parameters (beta, thetas)
rng = np.random.default_rng(12345)

sample_beta = rng.dirichlet(concentration_parameter,n_words)
sample_theta_0 = rng.dirichlet((alpha_0), int(.5*n_doc))
sample_theta_1 = rng.dirichlet((alpha_1), int(.5*n_doc))


# plt.barh(range(int(.5*n_doc)), sample_theta_1.transpose()[0])
# plt.barh(range(int(.5*n_doc)), sample_theta_1.transpose()[1], left=sample_theta_1.transpose()[0], color='g')
# plt.barh(range(int(.5*n_doc)), sample_theta_1.transpose()[2], left=sample_theta_1.transpose()[0]+sample_theta_1.transpose()[1], color='r')
# plt.title("Topic Distribution per sample document")
# plt.show()

# sample words
for doc in range(n_doc):
    p = sample_theta_0@sample_beta.T
    np.random.multinomial(40, p[doc], size = 1)
    