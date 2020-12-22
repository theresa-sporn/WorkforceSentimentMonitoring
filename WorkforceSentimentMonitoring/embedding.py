from WorkforceSentimentMonitoring.data import get_data, merge, drop_wrong_language
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
import numpy as np
import swifter
import pickle
import os
import numpy as np
import pandas as pd

embedder = DocumentPoolEmbeddings([FlairEmbeddings('news-forward-fast'),
                                   FlairEmbeddings('news-backward-fast')])
def embed(text, embedder):
    sentence = Sentence(text)
    embedder.embed(sentence)
    return sentence.get_embedding().detach().numpy()


if __name__ == '__main__':
    # submission, train, test = get_data()
    # df = merge(submission, train, test)
    # df = drop_wrong_language(df, "review")

    # path = os.path.split(os.path.abspath('__file__'))[0]
    # file = os.path.join(path, 'pickle_files/reviews_eng.p')
    # with open(file, 'wb') as f:
    #     pickle.dump(df, f)
    path = os.path.split(os.path.abspath('__file__'))[0]
    file = os.path.join(path, 'pickle_files/reviews_eng.p')
    with open(file, 'rb') as f:
        df = pickle.load(f)
    print(df)

    # X_tmp, y_tmp = (X.sample(1, random_state=21).reset_index(drop=True),
    #                y.sample(1, random_state=21).reset_index(drop=True))
    
    chunks = np.array_split(df, 10)
    for idx, chunk in enumerate(chunks, 1):
        print(type(chunk))
        chunk['embedding'] = chunk['review'].swifter.allow_dask_on_strings()\
                             .apply(lambda x: embed(x, embedder))
        file = os.path.join(path, f'pickle_files/embeddings_chunk{idx}')
        with open(file, 'wb') as f:
            pickle.dump(chunk, f)

    

    # path = os.path.split(os.path.abspath('__file__'))[0]
    # file = os.path.join(path, 'pickle_files/embeddings_all.p')
    # with open(file, 'wb') as f:
    #     pickle.dump(df, f)
