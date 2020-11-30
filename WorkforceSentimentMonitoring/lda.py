#step 1: get model output
#step 2: per column (topics 1-5 + overall) make 3 dfs: positive (score >3), negative (score <3), neutral (score = 3)
#step 3: calculate percentage positive, negative, neutral + display pie chart
#step 4: make pyLDAvis out of df positive and df negative:
    #a:

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import pprint as pp
import pyLDAvis
import pyLDAvis.gensim


def get_negatives(text):
    negatives = df['overall_score'] <3
    return negatives

def get_positives(text):
    positives = df['overall_score'] >3
    return positives

def get_neutrals(text):
    neutrals = df['overall_score'] == 3
    return neutrals

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):
    id2word = corpora.Dictionary(text)
    texts = tokenized
    corpus = [id2word.doc2bow(text) for text in texts]
    ldamallet = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=2, id2word=id2word, iterations=100)
    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    #pp.pprint(ldamallet.show_topics(formatted=False))
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, alpha=.91)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def make_pyldavis(dictionary, corpus, texts, start, limit, step):
    dictionary = id2word
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=4, limit=16, step=2)
    model_list.index(x.index(max(coherence_values))) #change!
    max_y = max(coherence_values)
    max_x = coherence_values.index(max(coherence_values))  #change!
    optimal_model = model_list[coherence_values.index(max(coherence_values))] #change!
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(optimal_model, corpus, dictionary=optimal_model.id2word)
    return vis
