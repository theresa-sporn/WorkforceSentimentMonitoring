import gemsim
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
import pyLDAvis
import pyLDAvis.gensim

def extract_negative(df):
    """return df with negative reviews and their labels"""
    negatives = df[['negatives']]
    return negatives

def extract_positive(df):
    """return df with negative reviews and their labels"""
    positive = df[['positives']]
    return positives


id2word = corpora.Dictionary(negatives_tokenized)
dictionary = id2word
dictionary = dictionary.filter_extremes(no_above=0.80)

texts = negatives_tokenized
corpus = [id2word.doc2bow(text) for text in texts]

ldamallet = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=2, id2word=id2word, iterations=100)

coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()


def compute_coherence_values(dictionary, corpus, texts, limit=8, start=2, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, alpha=.91)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=4, limit=8, step=2)

max_y = max(coherence_values)  # Find the maximum y value
max_x = coherence_values.index(max(coherence_values))  # Find the x value corresponding to the maximum y value
#xmax = x[numpy.argmax(y)]

optimal_model = model_list[coherence_values.index(max(coherence_values))]

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(optimal_model, corpus, id2word)
vis

