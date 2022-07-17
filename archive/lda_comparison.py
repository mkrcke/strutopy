from time import time
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from generate_docs import CorpusCreation

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.DEBUG)
logger = logging.getLogger("comp_clean")

gs_single_core = True
test_mode = 'online'

max_e_steps = 100

n_docs = 10000
n_test = 1000
n_topics = 10

n_features = 2000

# n_docs_per_job = 10000
n_jobs = 1

n_passes = 5
max_iterations = n_passes

kappa = 0.5
tau0 = 1.

eval_every = 1

# Train Test Split
train = Corpus.documents[:int(.8*len(Corpus.documents))]
test = Corpus.documents[int(.8*len(Corpus.documents)):]

def load_dataset():
    train = fetch_20newsgroups(
        subset='train',
        random_state=1,
        remove=(
            'headers',
            'footers',
            'quotes')).data[
        0:n_docs]
    test = fetch_20newsgroups(
        subset='test',
        random_state=1,
        remove=(
            'headers',
            'footers',
            'quotes')).data[
        0:n_test]
    return train, test


def sklearn_run(test_mode=None,
                train_X=None,
                test_X=None,
                n_topics=None,


                n_jobs=None,

                max_iterations=None,
                vectorizer=None,
                decay=None,
                offset=None,
                total_samples=None,
                batch_size=None

                ):

    logger.info("=================starting sklearn==================")

    alpha = None
    eta = None

    # difference with gensim. in gensim and hoffmann first iteration is 0,
    # while in sklearn it is 1
    offset = offset - 1
    # sklearn
    lda_sklearn = LatentDirichletAllocation(
        n_topics=n_topics,
        doc_topic_prior=alpha,
        topic_word_prior=eta,
        batch_size=batch_size,
        learning_decay=decay,
        learning_offset=offset,
        n_jobs=n_jobs,
        total_samples=total_samples,
        random_state=0,
        verbose=1,
        max_iter=max_iterations,
        learning_method=test_mode,
        max_doc_update_iter=max_e_steps,
        evaluate_every=eval_every)
    logger_sk = logging.getLogger('sklearn.online_lda')
    logger_sk.setLevel(logging.DEBUG)

    print('run test in %s mode' % test_mode)

    t0 = time()
    if test_mode == 'batch':
        # for batch mode
        lda_sklearn.fit(train_X)
    else:
        if n_passes == 1:

           # for online mode
            lda_sklearn.partial_fit(train_X)
        else:
            lda_sklearn.fit(train_X)

    sk_time = (time() - t0)
    print("sklearn fit in %0.3fs." % sk_time)

    # transform
    train_gamma = lda_sklearn.transform(train_X)
    train_perplexity = lda_sklearn.perplexity(train_X, train_gamma)

    test_gamma = lda_sklearn.transform(test_X)

    test_perplexity = lda_sklearn.perplexity(test_X, test_gamma)

    print('sklearn preplexity: train=%.3f, test=%.3f' %
          (train_perplexity, test_perplexity))

    return (train_perplexity, test_perplexity, sk_time)


def pre_processing(
        train_data=None,
        test_data=None,
        max_features=None,
        max_df=0.8,
        min_df=3):

    # sklearn format
    vectorizer = CountVectorizer(max_df=max_df, max_features=max_features,
                                 min_df=min_df, stop_words='english')
    train_X = vectorizer.fit_transform(train_data)
    test_X = vectorizer.transform(test_data)

    return (train_X, test_X, vectorizer)

def run_all(load_data=None):
    train_data, test_data = load_data()

    train_X, test_X, vectorizer = pre_processing(
        train_data=train_data, test_data=test_data, max_features=n_features)

    sklearn_results = sklearn_run(
        test_mode=test_mode,
        train_X=train_X,
        test_X=test_X,
        n_topics=n_topics,
        n_jobs=n_jobs,
        max_iterations=max_iterations,
        vectorizer=vectorizer,
        decay=kappa,
        offset=tau0,
        total_samples=n_docs,
        batch_size=sk_batch_size)


    print('n_jobs = %s', n_jobs)
    print('sklearn preplexity: train=%.3f, test=%.3f in %3f seconds',
          sklearn_results[0], sklearn_results[1], sklearn_results[2])


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(topic)
        #print("normalised:", topic / topic.sum()
        print(" ".join([feature_names[i] + '#' + str(i)
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

    print('all topics')
    print(model.components_)


def main():
    run_all(load_data=load_dataset)


if __name__ == '__main__':
    main()
