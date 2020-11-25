# merging and cleaning of dataset

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def get_data():

    # paths to raw_data folder containing .csv files
    # get data from .csv files (relative paths)

    path = os.path.join(os.getcwd(), '../raw_data')
    print(os.path.join(path, 'sample_submission.csv'))
    submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    test = pd.read_csv(os.path.join(path,'test.csv'))

    return submission, train, test


def merge(submission, train, test):

    # get clean dataframe


    # merge dataframes and drop unnecessary columns
    test = pd.merge(test, submission, on=['ID'])
    frames = [train, test]
    df = pd.concat(frames)
    df = df.rename(columns={'score_1':'work-balance', 'score_2':'culture-values', 'score_3':'career-opportunities', 'score_4':'comp-benefits', 'score_5':'senior-mgmt', 'score_6':'helpful-count'})
    df = df.drop(columns=['ID', 'Place', 'location', 'date', 'status', 'job_title', 'helpful-count'])


    # create a review column containing all text information
    text_columns = ['summary', 'positives', 'negatives', 'advice_to_mgmt']
    df['review'] = df[text_columns].astype('U').agg(' '.join, axis=1)


    # drop missing values
    categories = ['work-balance', 'culture-values', 'career-opportunities', 'comp-benefits', 'senior-mgmt', 'overall']
    df = df.dropna(axis=0, subset=categories)
    df = df.drop_duplicates()
    df[categories] = df[categories].astype('uint8')

    return df


def holdout(df, target):

    y = df[f'{target}']
    X = df.drop(f'{target}', axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

    return (X_train, X_val, y_train, y_val)


if __name__ == '__main__':

    df = get_data()
    print(df)
