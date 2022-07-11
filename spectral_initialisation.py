
#%%
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from qpsolvers import solve_qp
from scipy.sparse import bsr_array, bsr_matrix, diags

def spectral_init():
    """
    The default 
    choice, \code{"Spectral"}, provides a deterministic initialization using the
    spectral algorithm given in Arora et al 2014.  See Roberts, Stewart and
    Tingley (2016) for details and a comparison of different approaches.
    Particularly when the number of documents is relatively large we highly
    recommend the Spectral algorithm which often performs extremely well.  Note
    that the random seed plays no role in the spectral initialization as it is
    completely deterministic (unless using the \code{K=0} or random projection
    settings). When the vocab is larger than 10000 terms we use only the most
    frequent 10000 terms in creating the initialization.  This may case the 
    first step of the algorithm to have a very bad value of the objective function
    but it should quickly stabilize into a good place.  You can tweak the exact 
    number where this kicks in with the \code{maxV} argument inside control. There
    appear to be some cases where numerical instability in the Spectral algorithm
    can cause differences across machines (particularly Windows machines for some reason).
    It should always give exactly the same answer for a given machine but if you are
    seeing different answers on different machines, see https://github.com/bstewart/stm/issues/133
    for a longer explanation. 
    
    Input: 
    - documents in a bag of words format
    - list of anchor words of dimension K 
    Returns:
    - word-word-co-occurence matrix of dimension V x V
    """ 
    pass

#%% load example data 
def gram(doc_term_matrix): 
    """"Take a Matrix object and compute a gram matrix
        
        Due to numerical error in floating points you can occasionally get
        very small negative values.  Thus we check if the minimum is under 0 
        and if true, assign elements less than 0 to zero exactly.  This is mostly
        so we don't risk numerical instability later by introducing negative numbers of 
        any sort.

    Args:
        mat (@param): A Matrix sparse Document by Term matrix (D x V)

    Returns:
        _type_: _description_
    """
    word_counts = doc_term_matrix.sum(axis=1)  # absolute word counts per document
    # doc_term_matrix= doc_term_matrix[word_counts>=2,] # if word v occurs less than twice
    # word_counts = word_counts[word_counts>=2] 
    divisor = np.array(word_counts)*np.array(word_counts-1)
    # convert to sparse matrices to save some time
    doc_term_matrix = bsr_matrix(doc_term_matrix)
    Htilde = bsr_matrix(doc_term_matrix/np.sqrt(divisor))
    Hhat = diags(np.array(np.sum(doc_term_matrix / divisor, axis=0)).flatten(),0)
    Q = Htilde.T@Htilde - Hhat
    # Q = (doc_term_matrix/(np.sqrt(divisor))).T @ (doc_term_matrix/(np.sqrt(divisor))) - Hhat
    # Q is V x V matrix 
    Qsums = np.sum(Q, axis=1)
    return Q / Qsums

def fastAnchor(Qbar, K, verbose=True):
    """Find Anchor Words

    Take a gram matrix Q and returns K anchors.

    @input Q The gram matrix
    @input K The number of desired anchors
    @input verbose If True prints a dot to the screen after each anchor

    Returns:
        _type_: _description_
    """
    rowSquaredSums = np.sum(Qbar**2, axis=0) #StabilizedGS
    basis = np.zeros(K)
    for i in range(K):
        basis[i] = np.argmax(rowSquaredSums) #83-94

        maxval = rowSquaredSums[int(basis[i])]
        normalizer = 1/np.sqrt(maxval) #100
        
        #101-103
        Qbar[int(basis[i]),] = Qbar[int(basis[i]),]*normalizer 
        
        #For each row
        innerproducts = np.matmul(Qbar,Qbar[int(basis[i]),]) #109-113
        
        #Each row gets multiplied out through the Basis
        project = np.outer(innerproducts,Qbar[int(basis[i]),])

        #Now we want to subtract off the projection but
        #first we should zero out the components for the basis
        #vectors which we weren't intended to calculate
        project[np.int0(basis),] = 0 #106 (if statement excluding basis vectors)
        Qbar = Qbar - project #119
        rowSquaredSums[np.int0(basis)] = 0 
        #here we cancel out the components we weren't calculating.
        rowSquaredSums = np.sum(Qbar, axis=1)
        if verbose: print('.', end='', flush=True)
    return basis


def recover_l2(Qbar, anchor, wprob, verbose=True):
    """ Recover the topic-word parameters from a set of anchor words using the RecoverL2
        procedure of Arora et. al.

    Using the exponentiated algorithm and an L2 loss identify the optimal convex 
    combination of the anchor words which can reconstruct each additional word in the 
    matrix.  Transform and return as a beta matrix.

    @input Qbar The row-normalized gram matrix
    @input anchor A vector of indices for rows of Q containing anchors
    @input wprob The empirical word probabilities used to renorm the mixture weights. 
    @input verbose If TRUE prints information as it progresses.
    @input ... Optional arguments that will be passed to the exponentiated gradient algorithm.
        @return 

    Returns:
        A (np.ndarray): word-topic matrix of dimension K by V
    """
    M = Qbar[np.int0(anchor),]
    P = np.dot(M.T,M)

    #In a minute we will do quadratic programming
    #these jointly define the conditions.  First column
    #is a sum to 1 constraint.  Remainder are each parameter
    #greater than 0.
    G = np.eye(M.shape[0], M.shape[0])
    h = np.zeros(M.shape[0])
    A = np.ones(M.shape[0])
    #Amat = np.hstack((np.ones(X.shape[0])[:, np.newaxis], np.eye(X.shape[0], X.shape[0])))
    b = np.zeros(M.shape[0])
    b[0] = 1
    # Word by Word Solve for the Convex Combination 
    condprob = np.empty(Qbar.shape[0], dtype=np.ndarray, order='C')
    print("Shapes:")
    print("M:",M.shape)
    print("G:",G.shape)
    print("A:",A.shape)
    print("P:",P.shape)
    print("h:",h.shape)
    print("b:",b.shape)
    print("y:", Qbar[0,].shape)
    for i in range(len(Qbar)):
        if i in anchor: 
            vec = np.repeat(0,P.shape[0])
            vec[np.where(anchor==i)] = 1
            condprob[i] = vec
        else:
            y = Qbar[i,]
            q = y.T*M
            #meq=1 means the sum is treated as an exact equality constraint
            #and the remainder are >=
            solution = solve_qp(P=P, q=q, G=G, h=h,
                                       A=A, b=b, solver='quadprog')
        # if not np.any(solution): 
        #     solution[solution==np.nan] += 1e-10
    
        condprob[i] = solution

    # weights = np.vstack(condprob)
    # A = weights*wprob
    # A = A.T / np.sum(A, axis=0)
    return condprob

            



#%% raw documents
data = pd.read_csv('data/poliblogs2008.csv')
# selection for quick testing (make sure it is in line with R selection)
data = data[:1000]
# use the count vectorizer to get absolute word counts
vectorizer = CountVectorizer()
doc_term_matrix = vectorizer.fit_transform(data.documents)
#%% 
wprob = np.sum(doc_term_matrix, axis=1)
wprob = wprob/np.sum(wprob) 

beta = recover_l2(Q, anchor, wprob, verbose=True)

# %%
from numpy import array, dot
import qpsolvers
M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = dot(M.T, M)  # quick way to build a symmetric matrix
q = dot(array([3., 2., 3.]), M).reshape((3,))
G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = array([3., 2., -2.]).reshape((3,))
A = array([1., 1., 1.])
b = array([1.])

x = solve_qp(P, q, G, h, A, b, solver="quadprog")
print(f"QP solution: x = {x}")
# %%

# shapes (30729,) and (10,30729) not aligned: 30729 (dim 0) != 10 (dim 0)

M_test = np.array(([1,2,3,4],[1,2,3,4],[1,2,3,4]))
test=np.array([1,2,3])
np.dot(test, M_test.T)
# %%
