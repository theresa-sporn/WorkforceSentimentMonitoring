# merging and cleaning of dataset

import pandas as pd
from sklearn.model_selection import train_test_split
import os
from langdetect import detect
from WorkforceSentimentMonitoring.utils import timing, progress_bar
import time

SCORE_COLS = [
    "work-balance",
    "culture-values",
    "career-opportunities",
    "comp-benefits",
    "senior-mgmt",
    "overall",
]


def get_data():

    path = os.path.split(os.path.abspath(__file__))[0]
    path_to_data = os.path.join(path, "../raw_data")

    submission = pd.read_csv(os.path.join(path_to_data, "sample_submission.csv"))
    train = pd.read_csv(os.path.join(path_to_data, "train.csv"))
    test = pd.read_csv(os.path.join(path_to_data, "test.csv"))

    return submission, train, test


def merge(submission, train, test):

    # get clean dataframe

    # merge dataframes and drop unnecessary columns
    test = pd.merge(test, submission, on=["ID"])
    frames = [train, test]
    df = pd.concat(frames)
    df = df.rename(
        columns={
            "score_1": "work-balance",
            "score_2": "culture-values",
            "score_3": "career-opportunities",
            "score_4": "comp-benefits",
            "score_5": "senior-mgmt",
            "score_6": "helpful-count",
        }
    )
    df = df.drop(
        columns=[
            "ID",
            "Place",
            "location",
            "date",
            "status",
            "job_title",
            "helpful-count",
        ]
    )

    # create a review column containing all text information
    text_columns = ["summary", "positives", "negatives", "advice_to_mgmt"]
    df["review"] = df[text_columns].fillna('').agg(' '.join, axis=1)

    # drop missing values
    categories = [
        "work-balance",
        "culture-values",
        "career-opportunities",
        "comp-benefits",
        "senior-mgmt",
        "overall",
    ]
    df = df.dropna(axis=0, subset=categories)
    df = df.drop_duplicates()
    df[categories] = df[categories].astype("uint8")

    return df


def holdout(df, target):

    y = df[target]
    X = df.drop(target, axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

    X_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)

    return (X_train, X_val, y_train, y_val)

@progress_bar(expected_time=480)
def detect_wrong_language(df, column, language='en'):
    '''returns boolean series'''
    return df[column].apply(detect) != language

def drop_wrong_language(df, column, language = 'en'):
    '''drops entries written in languages other thatn the specified'''
    print('Identifying entries in other languages...')

    is_wrong = detect_wrong_language(df, column, language)

    n_rows_to_drop = is_wrong.sum()

    if n_rows_to_drop == 0:
        print('No entries to drop.')
        return df

    user_confirmation = None
    while not (user_confirmation is 'y' or user_confirmation is 'n'):
        user_confirmation = input(f'Drop {n_rows_to_drop} entries? [y] / n\n') or 'y'
    if user_confirmation is 'y':
        print(f'Dropping {n_rows_to_drop} entries...')
        df = df[~is_wrong]
        df.reset_index(inplace=True, drop=True)
        return df
    else:
        print('Process aborted')
        return df

def encode_target(y):
    encoding = {1 : 0, 2 : 0, 3 : 0, 4 : 1, 5 : 1}
    for col in y.columns:
        y[col] = y[col].map(encoding)

    return y

def get_prepaired_data(target=SCORE_COLS, keep_text_cols=False):
    """runs all functions above and returns X & y datasets (train & test) ready for preprocessing
       if keep_text_cols=True the returned DataFrame will keep the text columns from which the review column is created"""
    # retrieve data
    print('Reading data...')
    submission, train, test = get_data()
    # merge data
    print('Merging data into a single DataFrame...')
    df = merge(submission, train, test)
    # drop text columns if keep_text_cols = False
    if not keep_text_cols:
        print('Dropping initial text columns...')
        df = df.drop(columns=["summary", "positives", "negatives", "advice_to_mgmt"])
    # drop entries in wrong languages
    df = drop_wrong_language(df, 'review')
    # holdout
    print('Splitting train and test...')
    X_train, X_test, y_train, y_test = holdout(df, target)
    # Encode y_train and y_train
    print('Encoding targets...')
    y_train = encode_target(y_train)
    y_test = encode_target(y_test)
    print('Done!')

    return X_train, X_test, y_train, y_test
# are you working?

def get_lexicon():
    '''retrieves lexicon dictionary and loads it into a DataFrame'''
    lexicon = pd.read_csv('../lexicon/EmotionIntensityLexicon.txt',sep='\t')
    return lexicon

if __name__ == "__main__":


    X_train, X_test, y_train, y_test = get_prepaired_data()
