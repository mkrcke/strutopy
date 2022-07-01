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
#%% Train the model on the corpus.
test_topics = [2,3,5,10,20,50,100,110,120,130]

# Set training parameters.
for num_topics in test_topics: 
    chunksize = 20
    passes = 20
    iterations = 400
    eval_every = 5  # Don't evaluate model perplexity, takes too much time.
    lda = LdaModel(Corpus.documents, id2word = Corpus.dictionary, num_topics=num_topics, eval_every=eval_every)

    # evaluate the model
    top_topics = lda.top_topics(Corpus.documents)
    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print(f'Average topic coherence for {num_topics} topics: %.4f.' %avg_topic_coherence)
    

# %% 
# Set training parameters.
for num_topics in test_topics: 
    chunksize = 20
    passes = 20
    iterations = 400
    eval_every = 5  # Don't evaluate model perplexity, takes too much time.
    lda = LdaModel(Corpus.documents, id2word = Corpus.dictionary, num_topics=num_topics, eval_every=eval_every)

    # evaluate the model
    perplexity = lda.log_perplexity(Corpus.documents)

    print(f'perplexity for {num_topics} topics: %.4f.' %perplexity)

