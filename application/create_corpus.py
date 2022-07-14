# %%m imports


import pandas as pd
from gensim import corpora
from gensim.parsing.preprocessing import STOPWORDS

# %%

df = pd.read_csv("application/data/corpus_raw.csv", index_col=0)
text_corpus = list(df["text"])


# %% TODO: n-gramms , stemming,...
# all titles of docs as n-grams
# porter stemmer
# %% stopwords + split it by white space +  Lowercase each document

# Create a set of frequent words
stoplist = STOPWORDS.union(set("for a of the and to in he".split(" ")))

texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in text_corpus
]


# %% merge back to corpus_df

df["text_preproc"] = texts

# %% create word id dict

dictionary = corpora.Dictionary()
BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in texts]


# %% save objects
corpora.MmCorpus.serialize("application/data/BoW_corpus.mm", BoW_corpus)
dictionary.save("application/data/dictionary.mm")
df.to_csv("application/data/corpus_preproc.csv")

# %% how to reload the objects
corpora.dictionary.Dictionary.load("application/data/dictionary.mm")
corpora.MmCorpus("application/data/BoW_corpus.mm")
pd.read_csv("application/data/corpus_preproc.csv")
