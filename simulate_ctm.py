#%% import packages
import numpy as np
from generate_docs import CorpusCreation

from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

################# LDA #############################
# https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/ldamodel.py
# Usage examples
# --------------
# Train an LDA model using a Gensim corpus

#%% Simulate a corpus
np.random.seed(12345)
Corpus = CorpusCreation(
    n_topics = 10,
    n_docs=1000,
    n_words=100,
    V=5000,
    treatment=False,
    alpha='symmetric')

Corpus.generate_documents()
#sample unique tokens to build the dictionary
print('Number of unique tokens: %d' % len(Corpus.dictionary))
print('Number of documents: %d' % len(Corpus.documents))