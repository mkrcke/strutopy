# This script contains a class to simulate the data generating process for generative probabilistic text models.
# For this implementation, the documents are generated as it is described by Blei et al. (2003).
# Covariate effects can be introduced by specifying an average treatment effect (ATE) either as a linear effect (alpha_treatment="auto-linear") or as a nonlinear effect (alpha_treatment="auto-nonlinear")
# Specifying the treatment effect changes topic proportions of the prior distribution which is used to draw samples from. A linear effect can be used for two-level covariates.
# Using the nonlinear specification, the topic proportions change depending on a continuous covariate. Either case, one half of the simulated documents are sampled from the
# population that receives treatment and the other half is sampled from the population without treatment.


import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from gensim import utils
from gensim.corpora.dictionary import Dictionary

from archive.old_stm import stable_softmax

logger = logging.getLogger(__name__)


class CorpusCreation:
    """
    Class to simulate documents for topic model evaluation. For simulation,
    one can either use LDAs data generating process described by Blei et al. (2003), or
    STMs data generating process described by Roberts et al. (2016). As of now, the Corpus creation class
    is limited generating two boolean topical prevalence covariates. 

    Input
    ------------------------------------------------------------------------------------------------------------------
    - n_topics: {dtype: int}, required: number of topics - so far only n_topics=3 works
    - n_docs: {dtype: int}, required: number of documents to be generated
    - n_words: {dtype: int}, required: number of words for each document
    - V: {dtype: int}, required: length of the vocabulary to sample from
    - treatment: {dtype: bool}, optional
                    * True: Sampling procedure with a covariate effect on topic proportions of half of the documents
                    * False: (default) Sampling procedure without covariate effect on topic proportions
    - alpha: {dtype: float, list of float, str}, optional
                Concentration parameter for the document-topic distribution:
                    * scalar for a symmetric prior over document-topic distribution
                        - Example: `3.0`
                    * 1D array of length equal to num_topics to denote an asymmetric user defined prior for each topic.
                        - Example: 'np.array([.3,.4,.3])' for num_topics == 3
                Alternatively default prior selecting strategies can be employed by supplying a string:
                    * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`
                    * 'asymmetric': Uses a fixed normalized asymmetric prior of `1.0 / (topic_index + numpy.sqrt(num_topics))`,
                                    needs to have the same dimension as n_topics
    - alpha_treatment: {dtype: float, list of float, str}, required if treatment == True
                Concentration parameter for the document-topic proportion of the treatment:
                    * scalar for a symmetric prior over document-topic proportions for the treatment group
                    * 1D array of length equal to alpha to denote as user defined prior for each topic of the treatment
                Alternatively default prior settings for the treatment can be employed by supplying a string:
                    * 'auto-linear': Adjusts the prior by flipping proportions with `numpy.flip(alpha)`
                    * 'auto-nonlinear': Adjusts the prior by applying the exponential function with `numpy.exp(alpha)`

    Optionally, the user can provide topic-word-distributions and document-topic-distributions from a pre-trained model:

    @param: beta {dtype: ndarray}, optional: 2D-array providing topic-word-distributions with dimension KxV (global parameter)
    @param: theta {dtype: ndarray}, optional: 2D-array providing topic proportions with dimension NxK (document-level parameter)
    @param: level {dtype: int} count of topical prevalence covariates  

    Returns
    ------------------------------------------------------------------------------------------------------------------
    Corpus-like document representation
    A simulated text corpus as a bag-of-words. Each document is represented by a combination of word index and word count.
    Each tuple represents a document. Each document is represented by index-word pairs (tuples).

    dtype: list of tuples (Ex.: [((1,2)(5,1)(7,2)), ((3,1)(4,1)(9,2)), ((4,1)(6,1)(9,1))]

    Usage Example: Create synthetic corpus
    ------------------------------------------------------------------------------------------------------------------
    np.random.seed(12345)
    
    # Initialize corpus object
    Corpus = CorpusCreation(
        n_topics = 10,
        n_docs=1000,
        n_words=100,
        V=5000,
        dgp='STM',
        level=2, 
        )
    
    # Generate synthetic documents based on the data generating process
    Corpus.generate_documents()
    
    
    print('Number of unique tokens: %d' % len(Corpus.dictionary))
    print('Number of documents: %d' % len(Corpus.documents))

    
    """

    def __init__(
        self,
        n_topics,
        n_docs,
        n_words,
        V,
        level,
        treatment=False,
        alpha="symmetric",
        dgp="STM",
        metadata=None,
        alpha_treatment=None,
        beta=None,
        theta=None,
        gamma=None,
    ):

        # store corpus settings
        self.K = n_topics
        self.n_docs = n_docs
        self.n_words = n_words
        self.V = V
        self.dgp = dgp

        self.level = level

        # store prior parameters
        self.treatment = treatment

        # set seed
        self.rng = np.random.default_rng(12345)

        # store user supplied probability distributions for words and documents
        self.init_alpha(alpha, alpha_treatment, theta)
        self.word_topic_dist(beta)
        self.init_gamma(gamma)
        self.set_metadata(metadata)
        self.init_eta()
        self.init_theta(theta)

    def init_alpha(self, alpha, alpha_treatment, theta):
        if type(alpha) == np.ndarray:
            self.alpha = alpha
        else:
            if alpha == "symmetric":
                self.alpha = np.repeat((1 / self.K), self.K)
            elif alpha == "asymmetric":
                self.alpha = 1 / (
                    np.array(range(1, self.K + 1))
                    + np.sqrt(np.array(range(1, self.K + 1)))
                )
            else:
                self.alpha = np.repeat(alpha, self.K)

        if not np.any(self.alpha):
            assert (
                theta != None
            ), "Either alpha or theta needs to be specified for generating documents."

        if self.treatment == True:
            self.init_treatment(alpha_treatment)

    def init_treatment(self, alpha_treatment):
        assert (
            alpha_treatment is not None
        ), "If treatment == True, the effect needs to be specified by alpha_treatment"
        if type(alpha_treatment) == np.ndarray:
            self.alpha_treatment = alpha_treatment
        else:
            if alpha_treatment == "auto-linear":
                self.alpha_treatment = np.flip(self.alpha)
            elif alpha_treatment == "auto-nonlinear":
                self.alpha_treatment = np.exp(self.alpha)

    def word_topic_dist(self, beta):
        """2D numpy array containing the word-topic distribution

        Args:
            beta (nd.array): 2D-array of dimension K by V
        """
        if beta is None:
            self.beta = self.rng.dirichlet(size=self.K, alpha=np.repeat(0.05, self.V))
        else:
            self.beta = np.array(beta)
            assert type(self.beta) == np.ndarray, "beta needs to be a 2D numpy array"

    def init_gamma(self, gamma, mean=None):
        """if no value for gamma is provided, the values for k-topics are sampled from
        a p-dimensional multivariate normal distribution with mean and sigma.

        @param: gamma (optional) 2-dimensional np.ndarray containing normally distributed values"
        @param: mean (optional) prior mean for the topical prevalence coefficients. Defaults to None.

        @return: gamma vector of dimension (K-1) x p
        """
        if gamma is None:
            if mean is None:
                mean = np.random.standard_normal(self.level)
            sigma_prior = np.eye(self.level)
            mean = np.random.multivariate_normal(mean, sigma_prior)
            sigma = np.eye(self.level)
            self.gamma = np.random.multivariate_normal(mean, sigma, self.K - 1)
        else:
            self.gamma = gamma

    def set_metadata(self, metadata, metadata_levels=[0, 1], p=None):
        if metadata is None:
            # simulate one-hot encoding x1==True iff x2==False
            x1 = self.rng.choice(
                metadata_levels, size=int(self.n_docs), replace=True, p=None
            )
            x2 = self.rng.choice(
                metadata_levels, size=int(self.n_docs), replace=True, p=None
            )
            self.metadata = np.column_stack((x1, x2))
        else:
            assert metadata.shape == (
                self.n_docs,
                self.level,
            ), "Unexpected metadata shape provided."
            self.metadata = metadata
        return

    def init_eta(self):
        mu = self.metadata @ self.gamma.T
        sigma = np.eye(self.K - 1)
        eta = []
        for d in range(self.n_docs):
            eta.append(np.random.multivariate_normal(mu[d], sigma))
        self.eta = np.array(eta)
        return

    def init_theta(self, theta):
        """Generate documents for varying hyperprior on topical prevalence

        Example:
        if p = 2 and k=5, we need to draw a sample of 5 for each p
        result: how probable is each topic given p=1 and p=2 respectively.
        result: dimension of theta_k 5 x 1 (dim(theta) = 5 x 2)
        result: dimension of eta_k 4 x 1 (dim(eta) = 4 x 2)

        # instead of fixing the mean parameter for each topic, we choose a standard normal distribution
        # as of now, the covariance is fixed to zero and the variance is constant 1 over all topics

        @param: theta (optional) to specify a N x K dimensional document-topic distribution from a fitted model.
        """
        # theta is a document-level parameter, i.e. it does change across documents
        if self.dgp == "LDA":
            if theta is None:
                if self.treatment == False:
                    self.theta = self.rng.dirichlet(
                        alpha=self.alpha, size=self.n_docs
                    )  # Treatment effect (yes/no) Alpha (symmetric, asymmetric and so on...)
                else:
                    self.theta = self.rng.dirichlet(
                        alpha=self.alpha, size=int(self.n_docs / 2)
                    )
                    self.theta_treatment = self.rng.dirichlet(
                        alpha=self.alpha_treatment, size=int(self.n_docs / 2)
                    )
        elif self.dgp == "STM":
            self.map_eta(eta=self.eta)

        else:
            logger.WARNING('Not specified data generating process. Choose from "STM" or "CTM".')
            self.theta = np.array(theta)
            assert type(self.theta) == np.ndarray, "theta needs to be a 2D numpy array"

    def map_eta(self, eta):
        eta_ = np.array(list(map(lambda x: np.insert(x, len(x), 0), eta)))
        self.theta = np.array(list(map(lambda x: stable_softmax(x), eta_)))
        return

    def generate_documents(
        self, remove_terms=True, dictionary=True, display_props=False
    ):
        """_summary_
        Args:
            remove_terms (bool, optional): Want to remove terms that are not sampled? Also adjusts for the index. Defaults to True.
            dictionary (bool, optional): Want to create a gensim dictionary from the simulated corpus? Defaults to True.
            display_props (bool, optional): Want to display topic proportions per document? Defaults to False.
        """
        logger.info(f"Create corpus for K={self.K} topics.")
        self.sample_documents()

        if remove_terms:
            self.remove_infrequent_terms()
        if dictionary:
            logger.info("Create Dictionary from an simulated corpus.")
            self.dictionary()
        if display_props:
            self.display_props()

    def sample_documents(self):
        if self.dgp == "LDA":
            self.lda_probs()
        if self.dgp == "STM":
            self.p = self.theta @ self.beta
        self.documents = []
        for doc in range(self.n_docs):
            doc_words = self.rng.multinomial(self.n_words, self.p[doc], size=1)
            self.documents.append(
                list(
                    zip(
                        np.asarray(doc_words).nonzero()[1],
                        doc_words[np.where(doc_words > 0)],
                    )
                )
            )

    def lda_probs(
        self,
    ):
        """define probability vectors for the multinomial word draws"""
        # TO-DO: adjust probabilities based on nonlinear covariate effects
        if self.treatment == False:
            self.p = self.theta @ self.beta
        else:
            p = self.theta @ self.beta
            p_treatment = self.theta_treatment @ self.beta
            self.p = np.concatenate((p, p_treatment), axis=0)

    def remove_infrequent_terms(self):
        """
        returns documents reduced by all infrequent terms
        """
        wcountvec = np.concatenate(
            [np.repeat(x[0], x[1]) for sublist in self.documents for x in sublist]
        )
        wcounts = np.sort(np.unique(wcountvec))
        logger.info(f"removes {self.V-len(wcounts)} words due to no occurence")
        # create new indices using a mapping
        map_idx = list(zip(np.arange(len(wcounts)), wcounts))
        for i in range(len(self.documents)):
            old_idx = [tuple[0] for tuple in self.documents[i]]
            values = [tuple[1] for tuple in self.documents[i]]
            new_idx = [tuple[0] for tuple in map_idx if tuple[1] in old_idx]
            self.documents[i] = list(zip(new_idx, values))
        self.V = len(wcounts)

    def dictionary(self):
        # create ({dict of (int, str)}
        logger.info(f"Build dictionary of size {self.V}...")
        self.dictionary = Dictionary.from_corpus(self.documents)

    def display_props(self):
        """display topic proportions per document (works up to k=3)"""
        # theta is now stored in self.theta (and self.theta_treatment if treatment==True)

        # plt.barh(range(int(self.n_docs)), self.theta.transpose()[0], label="p(k=1)")

        plt.barh(
            range(int(self.n_docs)),
            self.theta.transpose()[1],
            left=self.theta.transpose()[0],
            color="g",
            label="p(k=2)",
        )
        plt.barh(
            range(int(self.n_docs)),
            self.theta.transpose()[2],
            left=self.theta.transpose()[0] + self.theta.transpose()[1],
            color="r",
            label="p(k=3)",
        )
        plt.title(
            f"Topic Distribution for {self.n_docs} sample documents ({self.K} topics)"
        )
        plt.legend()
        timestamp = str(time.asctime()[-13:-6]).replace(":", "_")
        plt.savefig(f"img/display_props_{timestamp}.png")
        plt.show()

    def split_corpus(self, validation_set=False, document_completion=True, proportion=0.8):
        assert type(self.documents) == list
        
        test_split_idx = int(proportion * len(self.documents))
        self.train_docs = self.documents[:test_split_idx]
            
        if validation_set:
            validate_split_idx = int(
                (proportion + (1 - proportion) / 2) * len(self.documents)
            )
            self.test_docs = self.documents[test_split_idx:validate_split_idx]
            self.validate_docs = self.documents[validate_split_idx:]

        else: 
            self.test_docs = self.documents[test_split_idx:]

        if document_completion:
            self.test_1_docs, self.test_2_docs = self.cut_in_half(self.test_docs)


    def cut_in_half(self, doc_set):
        """function to split a set of documents in two parts

        @param: doc_set (np.ndarray) set of documents in bag-of-words format

        @return: first_half returns the set with every other word removed (starting at index 0)
        @return: second_half returns the set with every other word removed (starting at index 1)
        """
        first_half = np.zeros(len(doc_set), dtype=np.ndarray)
        second_half = np.zeros(len(doc_set), dtype=np.ndarray)

        for doc in range(len(doc_set)):
            first_half[doc] = doc_set[doc][0::2]
            second_half[doc] = doc_set[doc][1::2]

        return first_half, second_half
