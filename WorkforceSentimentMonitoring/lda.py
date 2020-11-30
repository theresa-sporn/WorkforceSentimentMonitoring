import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import pprint as pp
import pyLDAvis
import pyLDAvis.gensim
from preprocessing import preprocessing
from utils import extract_negatives, extract_positives

# def get_negatives(text):
#     negatives = df['overall_score'] <3
#     return negatives

# def get_positives(text):
#     positives = df['overall_score'] >3
#     return positives

# def get_neutrals(text):
#     neutrals = df['overall_score'] == 3
#     return neutrals

text = preprocessing(text)
positives = extract_positives(text)
negatives = extract_negatives(text)
#texts = [positives, negatives]

def make_pyLDAvis(dictionary, corpus, texts, limit, start=2, step=1):
    id2word = corpora.Dictionary(tokenized)
    positives or negatives #change to taking preprocessed text from positive/negative df
    corpus = [id2word.doc2bow(text) for text in texts]
    ldamallet = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=2, id2word=id2word, iterations=100)
    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    #model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=4, limit=16, step=2)

    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, alpha=.91)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

    max_y = max(coherence_values)
    max_x = coherence_values.index(max(coherence_values))
    optimal_model = model_list[coherence_values.index(max(coherence_values))]

    #optimal_num_topics = coherence_values.index(max(coherence_values))
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(optimal_model, corpus, id2word)
    return vis
