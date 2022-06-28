# This script contains a class to simulate the data generating process for generative probabilistic text models. 
# For this implementation, the documents are generated as it is described by Blei et al. (2003).
# Covariate effects can be introduced by specifying an average treatment effect (ATE) either as a linear effect (alpha_treatment="auto-linear") or as a nonlinear effect (alpha_treatment="auto-nonlinear")
# Specifying the treatment effect changes topic proportions of the prior distribution which is used to draw samples from. A linear effect can be used for two-level covariates. 
# Using the nonlinear specification, the topic proportions change depending on a continuous covariate. Either case, one half of the simulated documents are sampled from the 
# population that receives treatment and the other half is sampled from the population without treatment. 


from random import betavariate
import numpy as np
import time
import matplotlib.pyplot as plt
from gensim.corpora.dictionary import Dictionary
import logging

logger= logging.getLogger(__name__)


class CorpusCreation:
    """
    Class to simulate documents for topic model evaluation. For simulation, LDAs data generating process described by Blei et al. (2003) is used:

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
    - alpha_treatment: {dtype: float, list of float, str}, optionally, only if treatment == True
                Concentration parameter for the document-topic proportion of the treatment:
                    * scalar for a symmetric prior over document-topic proportions for the treatment group
                    * 1D array of length equal to alpha to denote as user defined prior for each topic of the treatment
                Alternatively default prior settings for the treatment can be employed by supplying a string:  
                    * 'auto-linear': Adjusts the prior by flipping proportions with `numpy.flip(alpha)`
                    * 'auto-nonlinear': Adjusts the prior by applying the exponential function with `numpy.exp(alpha)`
    
    Optionally, the user can provide topic-word-distributions and document-topic-distributions from a pre-trained model: 
    - beta: {dtype: ndarray}, optional: 2D-array providing topic-word-distributions with dimension KxV (global parameter)
    - theta: {dtype: ndarray}, optional: 2D-array providing topic proportions with dimension NxK (document-level parameter)

    
    Returns
    ------------------------------------------------------------------------------------------------------------------
    Corpus-like document representation
    A simulated text corpus as a bag-of-words. Each document is represented by a combination of word index and word count. 
    Each tuple represents a document. Each document is represented by index-word pairs (tuples). 
    
    dtype: list of tuples

    Example: [((1,2)(5,1)(7,2)), ((3,1)(4,1)(9,2)), ((4,1)(6,1)(9,1))] 
    """

    def __init__(self, n_topics, n_docs, n_words, V, treatment=False, alpha='symmetric', alpha_treatment=None, beta=None, theta=None):

        # check for input variables

        # store corpus settings
        self.K = n_topics
        self.n_docs = n_docs
        self.n_words = n_words
        self.V = V

        # store prior parameters
        self.treatment = treatment
        
        # set seed
        self.rng = np.random.default_rng(12345)
    
        # store user supplied probability distributions for words and documents
        self.init_alpha(alpha, alpha_treatment)
        self.init_beta(beta)
        self.init_theta(theta)
    
    
    def init_alpha(self, alpha, alpha_treatment):
        if type(alpha) == np.ndarray:
            self.alpha = alpha
        else: 
            if alpha == 'symmetric':
                self.alpha = np.repeat((1/self.K), self.K)
            elif alpha == 'asymmetric':
                self.alpha = 1 / (np.array(range(1,self.K+1))+np.sqrt(np.array(range(1,self.K+1))))
            else: 
                self.alpha = np.repeat(alpha, self.K)
        
        if self.treatment == True:
            self.init_treatment(alpha_treatment)


    def init_treatment(self, alpha_treatment):
        assert alpha_treatment is not None
        if type(alpha_treatment)==np.ndarray: 
            self.alpha_treatment = alpha_treatment
        else: 
            if alpha_treatment=='auto-linear': 
                self.alpha_treatment = np.flip(self.alpha)
            elif alpha_treatment == 'auto-nonlinear': 
                self.alpha_treatment = np.exp(self.alpha)
            
    def init_beta(self, beta):
        # initialize beforehand: beta is a global parameter, i.e. it does not change across documents
        if beta == None:
            self.beta = self.rng.dirichlet(size=self.K, alpha=np.repeat(0.05, self.V))
        else: 
            self.beta = beta


    def init_theta(self, theta): 
        # theta is a document-level parameter, i.e. it does change across documents
        if theta == None: 
            if self.treatment == False: 
                self.theta = self.rng.dirichlet(alpha=self.alpha, size=self.n_docs) # Treatment effect (yes/no) Alpha (symmetric, asymmetric and so on...)
            else: 
                self.theta = self.rng.dirichlet(alpha=self.alpha, size=int(self.n_docs/2))
                self.theta_treatment = self.rng.dirichlet(alpha=self.alpha_treatment, size=int(self.n_docs/2))
        else: 
            self.theta = theta
    
    def generate_documents(self, remove_terms=True, dictionary=True, display_props=False):
        """_summary_
        Args:
            remove_terms (bool, optional): Want to remove terms that are not sampled? Also adjusts for the index. Defaults to True.
            dictionary (bool, optional): Want to create a gensim dictionary from the simulated corpus? Defaults to True.
            display_props (bool, optional): Want to display topic proportions per document? Defaults to False.
        """
        self.sample_documents()    

        if remove_terms:
            self.remove_infrequent_terms()
        if dictionary: 
            logger.info('Create Dictionary from an simulated corpus.')
            self.dictionary()
        if display_props: 
            self.display_props


    def sample_documents(self):
        self.get_probabilities()
        self.documents = []
        for doc in range(self.n_docs): 
            doc_words = self.rng.multinomial(self.n_words, self.p[doc], size=1)
            self.documents.append(list(zip(np.asarray(doc_words).nonzero()[1], doc_words[np.where(doc_words>0)])))
        
    def get_probabilities(self): #TO-DO: adjust probabilities based on nonlinear covariate effects
        if self.treatment == False: 
            self.p = self.theta@self.beta
        else: 
            p = (self.theta@self.beta)
            p_treatment = (self.theta_treatment@self.beta)
            self.p = np.concatenate((p, p_treatment), axis = 0)

    # def sample_documents(self):
        
    #     self.true_theta = np.empty(self.n_docs, dtype = object)
    
    #     # only iff treatment == True 
    #     # generate documents by drawing words from a multinomial with dirichlet priors  
    #     for doc in range(self.n_docs):
    #         # treatment assignment (first half of documents without treatment)
    #         if doc < 50:
    #             sample_theta_0 = self.rng.dirichlet(self.alpha)
    #             p = sample_theta_0@self.sample_beta
    #             # bookkeeping theta
    #             self.true_theta[doc] = sample_theta_0
    #         else:
    #             sample_theta_1 = self.rng.dirichlet(self.alpha_treatment)
    #             p = sample_theta_1@self.sample_beta
    #             # bookkeeping theta
    #             self.true_theta[doc] = sample_theta_1
    #         doc_words = np.random.multinomial(40, p, size = 1)
    #         # mimic corpus structure
    #         # going from np.array([1,0,0,1,0,2]) to  [(0,1),(3,1),(5,2)] for each document
    #         self.documents.append(list(zip(np.asarray(doc_words).nonzero()[1], doc_words[np.where(doc_words>0)])))
        
    def remove_infrequent_terms(self):
        """
        returns documents reduced by all infrequent terms
        """
        wcountvec = np.concatenate([np.repeat(x[0], x[1]) for sublist in self.documents for x in sublist])
        wcounts =  np.sort(np.unique(wcountvec))
        print(f'removes {self.V-len(wcounts)} words due to no occurence')
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
        logger.info(f'Creates a dictionary of size {self.V}...')
        self.dictionary = Dictionary.from_corpus(self.documents)
        # create a tuple mapping: [(1, word), (2, word), ...]
        # compare to dictionary object in gensim 
        
    # display topic proportions per document
    def display_props(self):
        # theta is now stored in self.theta (and self.theta_treatment if treatment==True)
        plt.barh(range(int(self.n_docs)), self.theta.transpose()[0], label='p(k=1)')
        plt.barh(range(int(self.n_docs)), self.theta.transpose()[1], left=self.theta.transpose()[0], color='g', label='p(k=2)')
        plt.barh(range(int(self.n_docs)), self.theta.transpose()[2], left=self.theta.transpose()[0]+self.theta.transpose()[1], color='r', label = 'p(k=3)')
        plt.title(f"Topic Distribution for {self.n_docs} sample documents ({self.n_topics} topics)")
        plt.legend()
        timestamp = str(time.asctime()[-13:-6]).replace(':','_')
        plt.savefig(f'display_props_{timestamp}.png')
        plt.show()
    





