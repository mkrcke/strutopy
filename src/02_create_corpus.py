
import os
import re
from pydoc import doc

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.parsing.preprocessing import STOPWORDS

artifacts_dir = "artifacts/wiki_data"

input_path = os.path.join(artifacts_dir, "wiki_corpus.csv")
stop_words_path = os.path.join("artifacts", "auxiliary_data", "stop_words_english.txt")


df = pd.read_csv(input_path, index_col=0)
df = df.drop(df.index[np.where((df["statistics"] == 1) & (df["ml"] == 1))[0]])

text_corpus = list(df["text"])

# remove punctuation
text_corpus = [re.sub(r"[^\w\s]", "", doc) for doc in text_corpus]

# remove numbers
text_corpus = ["".join([i for i in doc if not i.isdigit()]) for doc in text_corpus]

# Create a set of frequent words
with open(stop_words_path) as f:
    stoplist = f.read().split()

texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in text_corpus
]

df["text_preproc"] = texts

dictionary = corpora.Dictionary()
BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in texts]

corpora.MmCorpus.serialize("artifacts/wiki_data/BoW_corpus.mm", BoW_corpus)
dictionary.save("artifacts/wiki_data/dictionary.mm")
df.to_csv("artifacts/wiki_data/corpus_preproc.csv")
