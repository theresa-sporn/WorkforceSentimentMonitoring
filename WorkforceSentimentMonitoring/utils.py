import pandas as pd


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
