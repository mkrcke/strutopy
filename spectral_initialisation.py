import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from qpsolvers import solve_qp
from scipy.sparse import diags, csr_matrix, csr_array
from sklearn.preprocessing import normalize

def spectral_init(corpus, K, V, maxV=10000, verbose=True, print_anchor=False):
    """
    init='spectral' provides a deterministic initialization using the
    spectral algorithm given in Arora et al 2014.  See Roberts, Stewart and
    Tingley (2016) for details and a comparison of different approaches.
    The spectral initialisation is recommended if the number of documents is
    relatively large (e.g. D > 40.000). When the vocab is larger than 10000 terms
    only the most 10.000 frequent terms are used for initialisation. The maximum
    number of terms can be adjusted via the @param maxV. Note that the computation
    complexity increases on the magnitude of n^2.
    
    Numerical instabilities might occur (c.f. https://github.com/bstewart/stm/issues/133)
    
    @param: corpus in bag-of-word format -> [list of (int, int)]
    @param: K number of topics used for the spectral initialisation
    @param: (default=10000) maxV maximum number of most frequent terms used for spectral initialisation
    @param: verbose if True prints information as it progresses.
    @param: (default=False) print_anchor words from input documents

    @return: word-topic distribution obtained from spectral learning (K X V)
    """ 
    doc_term_matrix = create_dtm(corpus=corpus)

    wprob = np.sum(doc_term_matrix, axis=0)
    wprob = wprob/np.sum(wprob) 
    wprob = np.array(wprob).flatten()


    keep = np.argsort(-1*wprob)[:maxV]
    doc_term_matrix = doc_term_matrix[:,keep]
    wprob = wprob[keep]

    if verbose: print("Create gram matrix...")
    Q = gram(doc_term_matrix)
    
    if verbose: 
        print("Find anchor words...")

    anchor = fastAnchor(Q, K)
    if print_anchor: 
            for i,idx in enumerate(anchor): 
                print(f"{i}. anchor word: {vectorizer.get_feature_names_out()[np.int0(idx)]}")
    if verbose: 
        print('Recover values for beta')
    beta = recover_l2(Q, anchor, wprob)

    if keep is not None: 
        beta_new = np.zeros(K*V).reshape(K,V)
        beta_new[:,keep] = beta
        beta_new = beta_new + 0.001/V
        beta = beta_new/np.sum(beta_new)

    return beta

def create_dtm(corpus):
    """
    Create a sparse csr_matrix constructed from three arrays:
    
    - data[:] word counts: the entries of the matrix, in any order
    - i[:] document index: the row indices of the matrix entries
    - j[:] word index: the column indices of the matrix entries

    @param: corpus in bag-of-word format -> [list of (int, int)]
        corpus (list): list containing word indices and word counts per document

    @return: doc_term_matrix sparse matrix with document-term counts in csr format
    """

    word_idx = []
    for doc in corpus: 
        for word in doc:
            word_idx.append(word[0])
    word_idx = np.array(word_idx)

    doc_idx = []
    for i, doc in enumerate(corpus): 
        for word in doc:
            doc_idx.append(i)
    doc_idx = np.array(doc_idx)

    count = []
    for doc in corpus: 
        for word in doc:
            count.append(word[1])
    count = np.array(count)

    return csr_matrix((count, (doc_idx, word_idx)))

def gram(doc_term_matrix): 
    """"
    Computes a square matrix Q from the document term matrix. 
    Values of Q are row normalized. 
    
    Note: scipy.sparse matrix features are used to improve computation time
    
    @param: doc_term_matrix (D x V) in sparse csr format

    @return: sparse row-normalized matrix Q (VxV) in sparse csr format
    """
    # absolute word counts per document
    word_counts = doc_term_matrix.sum(axis=1)  
    
    #TODO: remove terms occuring less than twice
    # doc_term_matrix= doc_term_matrix[word_counts>=2,] 
    # word_counts = word_counts[word_counts>=2] 
    
    # convert to dense arrays for faster computation 
    divisor = np.array(word_counts)*np.array(word_counts-1)
    
    # convert back to sparse matrices to save some time
    doc_term_matrix = csr_matrix(doc_term_matrix)
    Htilde = csr_matrix(doc_term_matrix/np.sqrt(divisor))
    Hhat = diags(np.array(np.sum(doc_term_matrix / divisor, axis=0)).flatten(),0)
    
    # compute Q matrix (takes some time)
    Q = Htilde.T@Htilde - Hhat

    # normalize Q: 
    assert np.all(Q.sum(axis=1)>0), 'Encountered zeroes in Q row sums, can not normalize.'
    # row-normalise Q
    normalize(Q, copy=False)
    return Q

def fastAnchor(Q, K, verbose=True):
    """Find Anchor Words

    Take matrix Q and return an anchor term for each of K topics.
    Projection of all words onto the basis spanned by the anchors.

    Note: scipy.sparse matrix features are used to improve computation time

    @input: Q The gram matrix
    @input: K The number of desired anchors
    @input: verbose If True prints a dot to the screen after each anchor

    @return: anchor vector of indices for rows of Q containing anchors
    """
    # compute squared sum per row using scipy.sparse
    row_squared_sum = csr_array(Q.power(2).sum(axis=0))
    basis = np.zeros(K)

    for i in range(K):
        #most probable word over topics
        maxind = row_squared_sum.argmax() 
        basis[i] = maxind
        maxval = row_squared_sum.max()
        normalizer = 1/np.sqrt(maxval)
        
        #normalize the high probable word (maxind)
        Q[maxind] = Q[maxind]*normalizer
        
        #For each row
        innerproducts = Q@Q[maxind,].T 
        
        #Each row gets multiplied out through the Basis
        # (use numpy array as we are not gaining anything for sparse vectors)
        if i == 0: 
            project = innerproducts.toarray()@Q[maxind,].toarray()
        
        project = innerproducts@Q[maxind,]
        
        #Now we want to subtract off the projection but
        #first we should zero out the components for the basis
        #vectors which we weren't intended to calculate
        project[np.int0(basis),] = 0 
        Q = Q.A - project 

        # Q is not sparse anymore...
        row_squared_sum = np.sum(np.power(Q,2), axis=0)
        row_squared_sum[:,np.int0(basis)] = 0 
        if verbose: print('.', end='', flush=True)
    return basis


def recover_l2(Q, anchor, wprob):
    """
    Recover the topic-word parameters from a set of anchor words using the RecoverL2
    procedure of Arora et al.

    Using the exponentiated algorithm and an L2 loss identify the optimal convex 
    combination of the anchor words which can reconstruct each additional word in the 
    matrix. Transform and return as a beta matrix.

    @param: Q the row-normalized gram matrix
    @param: anchor A vector of indices for rows of Q containing anchors
    @param: wprob The empirical word probabilities used to renorm the mixture weights. 
         
    @return: A (np.ndarray): word-topic matrix of dimension K by V
    """
    # Prepare Quadratic Programming
    M = Q[np.int0(anchor),]
    P = np.array(np.dot(M,M.T).todense()) # square matrix 

    # # condition Ax=b coefficients sum to 1
    # A = np.ones(M.shape[0])
    # b = np.array([1])

    # conditino Gx >= h (-Gx >= 0 <=> Gx <= 0) coefficients greater or equal to zero 
    G = np.eye(M.shape[0], M.shape[0])
    h = np.zeros(M.shape[0])
    
    # initialize empty array to store solutions 
    condprob = np.empty(Q.shape[0], dtype=np.ndarray, order='C')
    # find convex solution for each word seperately:
    for i in range(Q.shape[0]):
        if i in anchor: 
            vec = np.repeat(0,P.shape[0])
            vec[np.where(anchor==i)] = 1
            condprob[i] = vec
        else:
            y = Q[i,]
            q = np.array((M@y.T).todense()).flatten()
            solution = solve_qp(
                P=P,
                q=q,
                G=G,
                h=h,
                verbose=True,
                # lower/upper bound for probabilities
                lb = np.zeros(M.shape[0]),
                ub = np.ones(M.shape[0]),
                solver = 'quadprog'
                )
            # replace small negative values with epsilon and store solution
            if np.any(solution<0): 
                solution[solution<0] = np.finfo(float).eps
            condprob[i] = solution
    
    # p(z|w)
    weights = np.vstack(condprob)
    # p(w|z) = p(z|w)p(w)
    A = weights.T*wprob
    # transform
    A = A.T / np.sum(A, axis=1)
    # check probability assumption
    assert np.any(A>0), 'Negative probabilities for some words.'
    assert np.any(A<1), 'Word probabilities larger than one.'
    return A.T
 

############ TESTING ######################
# data = pd.read_csv('data/poliblogs2008.csv')
# # selection for quick testing (make sure it is in line with R selection)
# data = data[:100]
# # use the count vectorizer to get absolute word counts
# vectorizer = CountVectorizer()
# doc_term_matrix = vectorizer.fit_transform(data.documents)
# K=10
# beta = spectral_init(doc_term_matrix, maxV=None, verbose=True, K=K)

