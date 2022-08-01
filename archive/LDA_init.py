from gensim.models import LdaModel
from generate_docs import CorpusCreation

## Generate documents
num_topics = 30
Corpus = CorpusCreation(n_topics=num_topics, n_docs=100, n_words=150, V=5000, treatment=False, alpha='symmetric')
Corpus.generate_documents()

                       
##  LDA initialization
model = LdaModel(Corpus.documents, id2word = Corpus.dictionary, num_topics=num_topics)
get_document_topics = [model.get_document_topics(item)[0][1] for item in Corpus.documents]

(model.get_document_topics(Corpus.documents[0])[0][1])