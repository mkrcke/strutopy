# The heldout likelihood is a measure of predictive performance
# Based on fitted parameter values, we want to maximise the probability of unseen word (or documents)
# The Likelihood is defined as P(W"|W',theta, beta)

# Computing the predictive likelihood for unseen documents requires reestimation (not necessarily)
# Computing the predictive likelihood for unseen words in a document used for training does not require reestimation 
# -> Document Completion Method

# Document Completion Approach
# 0. Estimate beta on the whole corpus
# Generate a test, training and validation set based on a corpus (80/10/10)
# 1. Remove a certain portion of each document (~50%)
# 2. Fit STM on the remaining words
# 3. Compute the likelihood for the test portion based on the trained model parameters (theta, beta) 
# 4. Average over words in a document to get the held-out likelihood for a single document
# 5. Average over each documents held-out likelihood

# Requirements. 
## STM.expectation_maximization() returns the document-level mean posterior values (theta, beta) - CHECK 
## A function that can split the corpus into test, train and validate
## We can introduce fitted parameters (theta, beta) into the document for computing the heldout-likelihood. 


#%% Load packages and model
import json
import matplotlib.pyplot as plt
import numpy as np 
from stm import STM
import pandas as pd
# load model
with open('stm_model_2.json') as f:
    stm_model = json.load(f)

#%% 80/10/10 split
corpus = stm_model['documents']
test_split_idx = int(.8*len(corpus))
validate_split_idx = int(.9*len(corpus)) 

#%% to array

train = np.array(corpus[:test_split_idx])
test = np.array(corpus[test_split_idx:validate_split_idx])
validate = np.array(corpus[validate_split_idx:])

#%% cut docs in half
def cut_in_half(set):
  """function to split a set of documents in two parts

  Args:
      set (np.ndarray): set of documents in bag-of-words format

  Returns:
      first_half: returns the set with every other word removed (starting at index 0)
      second_half: returns the set with every other word removed (starting at index 1)
  """
  first_half = np.empty(len(set), dtype=np.ndarray)
  second_half = np.zeros(len(set), dtype=np.ndarray)
  
  for doc in range(len(set)): 
    first_half[doc] = (set[doc][0::2])
    second_half[doc] = (set[doc][1::2])

  return first_half, second_half

test_1, test_2 = cut_in_half(test)
validate_1, validate_2 = cut_in_half(validate)

#%% retrain stm using train + half of test 
beta_train_corpus = np.concatenate((train, test))
theta_train_corpus = np.concatenate((train, test_1))
# update dictionary
from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary.from_corpus(train_corpus)

model_beta = STM(stm_model["settings"], beta_train_corpus, dictionary) 
model_theta = STM(stm_model["settings"], theta_train_corpus, dictionary)
model_beta.expectation_maximization(saving=False)
model_theta.expectation_maximization(saving=False)

#%% compute the likelihood for the remaining words in the test set
# take beta from model trained on train + test set
beta = np.array(model_beta.beta)
# take theta from model trained on train + test_1 set
theta = np.array(model_theta.lamda)

doc_ll=[]
for i,doc in enumerate(test_2):
  word_ll = []
  for word in doc:  
    word_ll.append(word[1]*np.log(theta[i]@beta[:,word[0]]))
  per_word_ll = np.sum(word_ll)/np.sum(np.array(doc)[:,1])
  doc_ll.append(per_word_ll)

print(doc_ll)
#%% Compute likelihood for second half of the test set
# somehow beta * theta
# for each document, take beta and theta and take the dot product
def compute_likelihood(document, beta, theta): 
    pass
#%%
make.heldout <- function(documents, vocab, N=floor(.1*length(documents)), 
                         proportion=.5, seed=NULL) {
  if(!is.null(seed)) set.seed(seed)
  
  # Convert the corpus to the internal STM format
  args <- asSTMCorpus(documents, vocab)
  documents <- args$documents
  vocab <- args$vocab

  index <- sort(sample(1:length(documents), N))
  pie <- proportion
  missing <- vector(mode="list", length=N)
  ct <- 0
  for(i in index) {
    ct <- ct + 1
    doc <- documents[[i]]  
    if(ncol(doc)<2) next
    doc <- rep(doc[1,], doc[2,])
    #how many tokens to sample? The max ensures at least one is sampled
    nsamp <- max(1,floor(pie*length(doc)))
    ho.index <- sample(1:length(doc), nsamp)
    tab <- tabulate(doc[ho.index])
    missing[[ct]] <- rbind(which(tab>0), tab[tab>0])
    tab <- tabulate(doc[-ho.index])
    documents[[i]] <- rbind(which(tab>0), tab[tab>0])
  }
  missing <- list(index=index, docs=missing)
  #check the vocab
  indices <- sort(unique(unlist(lapply(documents, function(x) x[1,]))))

  #all sorts of nonsense ensues if there is missingness
  #first condition checks the vocab, second checks the documents
  if(length(indices)!=length(vocab) | any(unlist(lapply(missing$docs, is.null)))) {
    remove <- which(!(1:length(vocab)%in% indices))
    newind <- rep(0, length(vocab))
    newind[indices] <- 1:length(indices)
    new.map <- cbind(1:length(vocab), newind)
    #renumber the missing elements and remove 0's
    missing$docs <- lapply(missing$docs, function(d) {
      d[1,] <- new.map[match(d[1,], new.map[,1]),2]
      return(d[,d[1,]!=0, drop=FALSE])
    })
    #apply the same process to the documents
    documents <- lapply(documents, function(d) {
      d[1,] <- new.map[match(d[1,], new.map[,1]),2]
      return(d[,d[1,]!=0, drop=FALSE])
    })
    
    lens <- unlist(lapply(missing$docs, length))
    if(any(lens==0)) {
      missing$docs <- missing$docs[lens!=0]
      missing$index <- missing$index[lens!=0]
    }
    vocab <- vocab[indices]
  }
  #hooray.  return some stuff.
  heldout <- list(documents=documents,vocab=vocab, missing=missing)

  #you can get cases where these come out as non-integers...
  #recast everything just to be sure.
  heldout$documents <- lapply(heldout$documents, function(x) matrix(as.integer(x), nrow(x), ncol(x)))
  heldout$missing$docs <- lapply(heldout$missing$docs, function(x) matrix(as.integer(x), nrow(x), ncol(x)))
  class(heldout) <- "heldout"
  return(heldout)
}
def make_heldout(): 
    """Function to split a portion of documents into training and testing parts
    as it is described in the Document Completion Method. 

    Arguments
    ----------------------------------------
    documents: corpus containing the documents to create the heldout sets for 
    vocab: 
    N (default: floor(.1*length(documents))
    proportion=.5
    seed=NULL

    Returns
    ----------------------------------------


    """
    pass

#%%

def evaluate_heldout(): 
    pass 
