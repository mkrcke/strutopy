#%%
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from modules.heldout import cut_in_half
from modules.stm import STM
import numpy as np
import pandas as pd 
from gensim.corpora.dictionary import Dictionary
from joblib import Parallel, delayed
from modules.chunk_it import chunkIt
from modules.heldout import eval_heldout
import logging
import multiprocessing
import os

import os
import re


# initialize logging
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    filename="artifacts/logfiles/application.log",
    encoding="utf-8",
    level=logging.INFO,
)
logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)

#%% define function for hyperparameter tuning 
def train_on_corpus(K, beta_train_corpus, theta_train_corpus, heldout_corpus):
    """
    1) Iterate over candidates for the number of topics K
          - choose K from [5,10,20,30,40,50]
    2) Fit the model on the training data (beta and theta separately)
          - a separate fit for theta and beta resembles the document completion approach.
          - create a corpora for training on the train and the train+first_test_half respectively.
          - models not necessarily need to converge, max EM iterations are set to 25
          - both model parameters are stored
    3) Evaluate the heldout likelihood on the test data
          - values obtained from step (2) are used to evaluate the heldout likelihood on second_test_half
          - likelihood is stored for comparing different model fits

    @param K (int, list of int): number of topic candidates used for hyperparameter optimisation
    @param beta_train_corpus (_type_): corpus for training
    @param theta_train_corpus (_type_): corpus for testing
    @param heldout_corpus (_type_): corpus for evaluation
    """
    stm_config = {
        "model_type": 'STM',
        "content": False,
        "K": K,
        "kappa_interactions": False,
        "sigma_prior": sigma_prior,
        "convergence_threshold": convergence_threshold,
        "init_type": "spectral",
    }

    # initialize dictionaries for different corpora
    model_beta_dictionary = Dictionary.from_corpus(beta_train_corpus)
    model_theta_dictionary = Dictionary.from_corpus(theta_train_corpus)

    # initialize models for theta and beta
    model_beta = STM(
        documents=beta_train_corpus,
        dictionary=model_beta_dictionary,
        X=X[: len(beta_train_corpus)],
        max_em_iter = max_em_iter,
        **stm_config,
    )

    model_theta = STM(
        documents=theta_train_corpus,
        dictionary=model_theta_dictionary,
        X=X[: len(theta_train_corpus)],
        **stm_config,
        max_em_iter = 25,
    )
    
    results_path=f"artifacts/wiki_data/v2/STM_{K}"
    os.makedirs(results_path, exist_ok=True)
    # Train model to retrieve beta and theta (document completion approach)
    logger.info(f"Fitting STM for K={K} ...")
    
    model_beta.expectation_maximization(saving=True, output_dir=results_path)
    model_theta.expectation_maximization(saving=True, output_dir=results_path)

    # Save Likelihood
    logger.info(f"Evaluate the heldout likelihood on the remaining words...")
    heldout_llh = eval_heldout(
        heldout_corpus, theta=model_theta.theta, beta=model_beta.beta
        )
    logger.info(f"Saving into {results_path}.")
    
    heldout_path= os.path.join(results_path,'heldout')
    np.save(heldout_path, np.array(heldout_llh))

#%% load text & preprocess
# ingest the corpus

artifacts_dir = "artifacts/wiki_data"

# ingest the corpus and the stopwords
input_path = os.path.join(artifacts_dir, "wiki_corpus.csv")
stop_words_path = os.path.join("artifacts", "auxiliary_data", "stop_words_english.txt")

# load data
data = pd.read_csv(input_path, index_col=0)
# df = df.drop(df.index[np.where((df["statistics"] == 1) & (df["ml"] == 1))[0]])

text_corpus = list(data["text"])

# remove punctuation
text_corpus = [re.sub(r"[^\w\s]", "", doc) for doc in text_corpus]

# remove numbers
text_corpus = ["".join([i for i in doc if not i.isdigit()]) for doc in text_corpus]

# Create a set of pre-defined stop words
with open(stop_words_path) as f:
    stoplist = f.read().split()

# add words to stoplist
custom_stopwords = [
    'statistics',
    'statistical',
    'data',
    'machine',
    'learning',
    'called',
    'displaystyle',
    "'",
    ]

for word in custom_stopwords:
    stoplist.append(word)

texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in text_corpus
]

data["text_preproc"] = texts

dictionary = corpora.Dictionary()
BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in texts]
#%% Prepare corpora for finding the optimal number of topics

# specify model parameters
kappa_interactions = False
beta_index = None
max_em_iter = 20
sigma_prior = 0
convergence_threshold = 1e-5

# set topical prevalence 
prevalence_covariate = ['statistics', 'ml']
X = data.loc[:, prevalence_covariate]

# split corpus based on 60/40 train-test split
TRAIN_TEST_PROPORTION = 0.6

test_split_idx = int(TRAIN_TEST_PROPORTION * len(BoW_corpus))

corpus = [doc for doc in BoW_corpus] 
train_docs = corpus[:test_split_idx]
test_docs = corpus[test_split_idx:]
test_1_docs, test_2_docs = cut_in_half(test_docs)

# Prepare corpora for model training
beta_train_corpus = np.concatenate([
    train_docs,
    test_docs,
])
theta_train_corpus = np.concatenate([
    train_docs,
    test_1_docs,
])
heldout_corpus = test_2_docs
 
# %% Fit the model for K candidates and save the results
#parallelized operation 
K_candidates = np.array([5,10,15,20,25,30])
cores_to_use = 2
# split according to maximal cores_to_use
t = K_candidates
t_split = chunkIt(t, float(len(t) / cores_to_use))
Parallel(n_jobs=len(t_split))(
    delayed(train_on_corpus)(candidate, beta_train_corpus, theta_train_corpus, heldout_corpus) for candidate in K_candidates
)
#%% Evaluate heldout likelihood for candidate models
def list_files(filepath, filetype):
   paths = []
   for root, dirs, files in os.walk(filepath):
      for file in files:
         if file.lower().endswith(filetype.lower()):
            paths.append(os.path.join(root, file))
   return(paths)

path = "artifacts/wiki_data/v2"
result_paths = list_files(path, 'heldout.npy')

values=[]
K=[]
for path in result_paths: 
    heldout = np.load(path)
    k_value = path.split('/')[-2].split('_')[-1]
    values.append(heldout)
    K.append(k_value)

result_frame = pd.DataFrame(
    {
        'K':K,
        'heldout':np.round(values,3)
    }
)

ax = sns.lineplot(data=result_frame.sort_values(by='K'), x="K", y="heldout")
ax.set(xlabel='Number of topics', ylabel='Per Word Heldout Likelihood')
plt.savefig('../img/llh_application', bbox_inches='tight', dpi=360)
# highest likelihood for K == 20 

#%% Evaluate convergence for candidate models
values=[]
K=[]
path = "artifacts/wiki_data/v2"
result_paths = list_files(path, 'lower_bound.pickle')
for i, path in enumerate(result_paths): 
    lb = np.load(path, allow_pickle=True)
    k_value = path.split('/')[-2].split('_')[-1]
    K.append(k_value)
    values.append(lb)

values[1] = values[1][:25]

for i in range(len(values)):
    print(i)
    k=K[i]
    plt.plot(range(25),[lb for lb in values[i]],label = 'K=%s'%k)
    

plt.legend()
plt.show()

#%%_____________________________________________________________________
## Fit Model for optimal choice of K obtained from above
# specify model parameters
kappa_interactions = False
lda_beta = True
beta_index = None
max_em_iter = 25
sigma_prior = 0
convergence_threshold = 1e-5

K=20

stm_config = {
        "model_type": 'STM',
        "content": False,
        "K": K,
        "kappa_interactions": False,
        "lda_beta": True,
        "sigma_prior": sigma_prior,
        "convergence_threshold": convergence_threshold,
        "init_type": "spectral",
    }
wiki_model = STM(
    documents=BoW_corpus,
    dictionary=dictionary,
    X=X,
    **stm_config,
    max_em_iter = 50,
)

wiki_model.expectation_maximization(saving=False)
#%% investigate advantage of spectral decomposition

lb_spectral = np.load('artifacts/wiki_data/final_model/lower_bound.pickle', allow_pickle=True)
lb_random = wiki_model.last_bounds # here wiki_model was estimated with "init_type": "random"



plt.plot(lb_random,label = 'random initialisation')
plt.plot(lb_spectral,label = 'spectral initialisation')
plt.legend()

plt.savefig('../img/spectral_initialisation', dpi=360, bbox_inches='tight')

# %% investigate topics (highest probable words)
K=10
prob, frex = wiki_model.label_topics(n=10, topics=range(K))
# investigate covariate effect on topics
for topic in range(K): 
    print(f"Statistics: {round(wiki_model.gamma[topic][0],4)} * {frex[topic]})")
    print(f"ML:  {round(wiki_model.gamma[topic][1],4)} * {frex[topic]} \n")
# %%
_,frex = wiki_model.label_topics(n=10, topics=range(0,10))
for i,topic in enumerate(frex):
    print('-'*130)
    print(f"Topic {i+1}:",topic[1:])

# %% investigate representative documents per topic
# The function find_thoughts() can be used to retrieve the most representative documents
# for particular, indexed topics. 
topics = [0]
for topic in topics: 
    print('Documents exhibiting topic "Testing"')
    titles = data['title'].iloc[wiki_model.find_thoughts(n=20, topics=[topic])]
    for i,title in enumerate(titles):
        print(f'document {i} {title}')
# %% investigate representative documents for (a single) topic 0 
topic = 5
data[['title','statistics','ml']].iloc[wiki_model.find_thoughts(n=25, topics=[topic])]
# %% topics with overlap
# to find topics that overlap, extract values where both gammas are of equal sign
stats_topics = np.where(wiki_model.gamma[:,0]>0)[0]
ml_topics = np.where(wiki_model.gamma[:,1]>0)[0]

# ML: label_topics
# Statistics: label_topics
for topic in ml_topics: 
    wiki_model.label_topics(n=15, topics =[topic], print_labels=True)
# Statistics: label_topics
for topic in stats_topics: 
    wiki_model.label_topics(n=15, topics =[topic], print_labels=True)

#%% for each topic, find representative documents
for topic in ml_topics:
    print('-'*25)
    print(np.array(data[['title']].iloc[wiki_model.find_thoughts(n=10, topics=[topic])]))

#%% Statistics
for topic in stats_topics:
    print('-'*25)
    print(np.array(data[['title']].iloc[wiki_model.find_thoughts(n=10, topics=[topic])]))

# %% distinct topics
# get largest absolute the distances in the covariate effects for five topics
#compute the absolute differences
gamma_diff = abs(wiki_model.gamma[:,1]-wiki_model.gamma[:,0])
distinct_topics = np.argpartition(gamma_diff, -5)[-5:]
# to find topics that are distinct, extract values where differences in gamma is largest
# label_topics
for topic in distinct_topics: 
    wiki_model.label_topics(n=15, topics =[topic], print_labels=True)
# find_thoughts

# %%
# wordclouds:
# 0) Generate a wordcloud for the entire data
# 1) extract representative documents for topic representatives
# 2) extract texts from the original data for the representatives
# 3) display words using wordcloud

# 1) generate a wordcloud for the whole corpus
x, y = np.ogrid[:300, :300]
flat_corpus = [token for sublist in data['text_preproc'] for token in sublist]
wc = WordCloud(max_words=1000, stopwords=stoplist, margin=5,
               random_state=1, background_color="white").generate(" ".join(flat_corpus))

plt.imshow(wc)
plt.axis("off")
plt.show()

# 2) generate a wordcloud for selected topics (2 x ML, 2 x Statistics)
##### STATISTICS CLOUDS #######
for topic in stats_topics:
    topic_texts = np.array(data['text_preproc'].iloc[wiki_model.find_thoughts(n=30, topics=[topic])])
    flat_topic_texts = [token for sublist in topic_texts for token in sublist]
    wc = WordCloud(max_words=1000, stopwords=stoplist, margin=5, mask=mask,
               random_state=1, background_color="white").generate(" ".join(flat_topic_texts))
    plt.imshow(wc)
    plt.axis("off")
    plt.savefig(f'../img/stats_{topic}', bbox_inches='tight', dpi=400)
    plt.show()

##### MACHINE LEARNING CLOUDS #######
for topic in ml_topics:
    topic_texts = np.array(data['text_preproc'].iloc[wiki_model.find_thoughts(n=30, topics=[topic])])
    flat_topic_texts = [token for sublist in topic_texts for token in sublist]
    wc = WordCloud(max_words=1000, stopwords=stoplist, margin=5, mask=mask,
               random_state=1, background_color="white").generate(" ".join(flat_topic_texts))
    plt.imshow(wc)
    plt.axis("off")
    plt.savefig(f'../img/ml_{topic}', bbox_inches='tight', dpi=400)
    plt.show()