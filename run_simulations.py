from simulate import generate_docs
import numpy as np


def basic_simulations(n_docs, n_words, V, ATE, alpha, display=True):
    generator = generate_docs(n_docs, n_words, V, ATE, alpha)
    documents = generator.generate(n_docs)
    if display == True:
        generator.display_props()
    return documents

documents = basic_simulations(n_docs=200, n_words=40, V=500, ATE=.2, alpha=np.array([.3,.4,.3]))

