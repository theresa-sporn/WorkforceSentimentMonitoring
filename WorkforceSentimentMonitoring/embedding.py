from WorkforceSentimentMonitoring.data import get_data, merge, drop_wrong_language
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
import numpy as np
import swifter
import pickle
import os

embedder = DocumentPoolEmbeddings([FlairEmbeddings('news-forward-fast'),
                                   FlairEmbeddings('news-backward-fast')])
def embed(text, embedder):
    sentence = Sentence(text)
    embedder.embed(sentence)
    return sentence.get_embedding().detach().numpy()


if __name__ == '__main__':
    submission, train, test = get_data()
    df = merge(submission, train, test)
    df = drop_wrong_language(df, "review")
    X = df[['review']]
    y = df.iloc[:, -7:-1]

    X_tmp, y_tmp = (X.sample(100, random_state=21).reset_index(drop=True),
                   y.sample(100, random_state=21).reset_index(drop=True))

    X_tmp['embedding'] = X_tmp['review'].swifter.allow_dask_on_strings()\
                         .apply(lambda x: embed(x, embedder))

    tmp = X_tmp.join(y_tmp)

    path = os.path.split(os.path.abspath('__file__'))[0]
    file = os.path.join(path, '../pickle_files/tmp.p')
    with open(file, 'wb') as f:
        pickle.dump(tmp, f)
