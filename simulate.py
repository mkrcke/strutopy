# This script contains the basic simulation functionalities

import numpy as np
import matplotlib.pyplot as plt


class generate_docs:
    """
    Class to simulate documents for topic model evaluation. For simulation, LDAs data generating process described by Blei et al. (2003) is used:

    Input
    ------------------------------------------------------------------------------------------------------------------ 
    - number of topics: n_topics (dtype: int) - so far only n_topics=3 works
    - number of documents to be generated: n_docs (dtype: int)
    - number of words for each document: n_words (dtype: int)
    - length of vocabulary: V (dtype: int)
    - covariate effect on the topic proportions: ATE (dtype: float)
    - concentration parameter for the Dirichlet distribution over topic proportions: alpha (dtype: numpy.array)
        example: 'np.array([.3,.4,.3])'
        needs to have the same dimension as n_topics
    - topic-word-distribution: symmetric Dirichlet distribution beta with dimension KxV (global parameter)
    - topic proportion: asymmetric Dirichlet distribution theta with dimension K (document-level parameter)
    Returns
    ------------------------------------------------------------------------------------------------------------------
    List of tuples, where each tuple represents a document. Each document is represented by index-word pairs (tuples). 

    Corpus-like document representation (BoW-representation)

    Example: [((1,2)(5,1)(7,2)), ((3,1)(4,1)(9,2)), ((4,1)(6,1)(9,1))] 
    """
    def __init__(self, n_docs, n_words, V, ATE, alpha, n_topics=3):
        

        self.alpha = alpha
        self.n_topics=n_topics


        
        self.n_docs = n_docs
        self.n_words = n_words
        self.V = V
        self.ATE = ATE
        self.t_1 = (-ATE,0,+ATE)

        
        # set seed
        self.rng = np.random.default_rng(12345)
        self.concentration_parameter = np.repeat(0.05, self.V)

        # sample corpus parameters (beta with dimension (KxV), theta with dimension (1xK))
        # beta is a global parameter, i.e. it does not change across documents -> sample beforehand
        # theta is a document-level parameter, i.e. it does change across documents -> sample during each iteration
        self.sample_beta = self.rng.dirichlet(size = self.n_topics, alpha= self.concentration_parameter)

    def generate(self, n_docs, remove_terms=True):
        # generate objects to fill
        self.documents = []
        self.true_theta = np.empty(n_docs, dtype = object)
        alpha_0 = self.alpha
        alpha_1 = np.add(alpha_0,self.t_1)
        # generate document by drawing words from a multinomial with dirichlet priors  
        for doc in range(n_docs):
            # treatment assignment (first half of documents without treatment)
            if doc < 50:
                sample_theta_0 = self.rng.dirichlet((alpha_0))
                p = sample_theta_0@self.sample_beta
                # bookkeeping theta
                self.true_theta[doc] = sample_theta_0
            else: 
                sample_theta_1 = self.rng.dirichlet((alpha_1))
                p = sample_theta_1@self.sample_beta
                # bookkeeping theta
                self.true_theta[doc] = sample_theta_1
            doc_words = np.random.multinomial(40, p, size = 1)
            # mimic corpus structure
            # going from np.array([1,0,0,1,0,2]) to  [(0,1),(3,1),(5,2)] for each document
            self.documents.append(list(zip(np.asarray(doc_words).nonzero()[1], doc_words[np.where(doc_words>0)])))
        if remove_terms:
            self.remove_infrequent_terms()
        vocabulary = self.V
        return self.documents, vocabulary

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
    
    # display topic proportions per document
    def display_props(self):
        # convert theta to array
        true_theta = np.vstack(self.true_theta)
        plt.barh(range(int(self.n_docs)), true_theta.transpose()[0], label='p(k=1)')
        plt.barh(range(int(self.n_docs)), true_theta.transpose()[1], left=true_theta.transpose()[0], color='g', label='p(k=2)')
        plt.barh(range(int(self.n_docs)), true_theta.transpose()[2], left=true_theta.transpose()[0]+true_theta.transpose()[1], color='r', label = 'p(k=3)')
        plt.title(f"Topic Distribution for {self.n_docs} sample documents ({self.n_topics} topics)")
        plt.legend()
        plt.show()


