import pandas as pd
import time

def extract_negative(df):
    """return df with negative reviews and their labels"""
    df = df[['negatives']]
    df.loc[:,'sentiment'] = 0 # 0=> negative
    return df

def extract_positive(df):
    """return df with positive reviews and their labels"""
    df = df[['positives']]
    df.loc[:,'sentiment'] = 1 # 1=> positive
    return df

# @decorator
def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed


def load_model():
    clf = joblib.load('model.joblib')
    print('model.joblib loaded')
    y_pred = clf.predict(X)
