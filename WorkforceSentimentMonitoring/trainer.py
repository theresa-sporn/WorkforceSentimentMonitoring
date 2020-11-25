import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB

from WorkforceSentimentMonitoring.data import get_data, merge, holdout
from WorkforceSentimentMonitoring.preprocessing import preprocessing

warnings.filterwarnings("ignore", category=FutureWarning)

FEATURE_COLS = ['summary', 'positives', 'negatives', 'advice_to_mgmt']
SCORE_COLS = ['score_1', 'score_2', 'score_3', 'score_4', 'score_5', 'overall']
def feature_engineer(df):
    # simple transformation from notebook, should be an extra feature package
    # iterate over features and append results to df as new cols
    result_scores = {}
    for feature in FEATURE_COLS:
        scores_dic = {}

        for score in SCORE_COLS:
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df[feature].astype('U'))
            y = df[score]
            model = MultinomialNB()
            model.fit(X, y)
            df[f'{feature}_{score}'] = model.predict(X)
            scores_dic[f'{score}'] = model.score(X, y)

        result_scores[f'{feature}'] = scores_dic

    # iterate over features and append results to df as new cols
    scores_dic = {}
    for score in SCORE_COLS:

        result_scores = {}
        for feature in FEATURE_COLS:
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df[feature].astype('U'))
            y = df[score]
            model = MultinomialNB()
            model.fit(X, y)
            df[f'{feature}_{score}'] = model.predict(X)
            result_scores[f'{feature}'] = model.score(X, y)

        scores_dic[f'{score}'] = result_scores
        scores_df = pd.DataFrame(scores_dic).T
    return df

# Create a function to get the subjectivity
def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity
def getPolarity(text):
  return TextBlob(text).sentiment.polarity

def feature_sentiment(df):

    for feature in FEATURE_COLS:
        df[f'subjectivity_{feature}'] = df[feature].astype('U').apply(getSubjectivity)
        df[f'polarity_{feature}'] = df[feature].astype('U').apply(getPolarity)
    return df

# Create a function to get the total length of the reviews
def get_lengths(df):
    '''returns a df with columns with the length of the reviews'''
    func = lambda x: len(x) if type(x) == str else 0
    df['summary_length'] = df['summary'].apply(func)
    df['postives_length'] = df['positives'].apply(func)
    df['negatives_length'] = df['negatives'].apply(func)
    df['advice_length'] = df['advice_to_mgmt'].apply(func)
    df['combined_length'] = df['text_combined'].apply(func)
    return df

def scaler(df):
    # scale new features
    length_cols = [col for col in df.columns if 'length' in col]

    for col in length_cols:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])

    # select X
    X = df.iloc[:, 11:]
    # scale score features
    pred_scores_cols = [col for col in X.columns if 'score' in col and not 'reg' in col]

    for col in pred_scores_cols:
        scaler = MinMaxScaler()
        X[col] = scaler.fit_transform(X[[col]])
    return X

def linear_feature(X, df):
    # linear regression with just the predictions for each model
    for col in SCORE_COLS:
        model = LinearRegression()
        model.fit(X, df[col])
        X[f'reg_{col}'] = model.predict(X)
        scaler = MinMaxScaler()
        X[f'reg_{col}'] = scaler.fit_transform(X[[f'reg_{col}']])
    return X


class Trainer(object):
    ESTIMATOR = "Logic"
    # SCORE_COLS = ['score_1', 'score_2', 'score_3', 'score_4', 'score_5', 'overall']
    # SCORE_COLS = ['work-balance', 'culture-values', 'career-opportunities', 'comp-benefits', 'senior-mgmt', 'overall']

    def __init__(self, X, y, **kwargs):
        self.kwargs = kwargs
        self.X = X
        self.y = y
        # self.df = self.get_lengths()
        # self.nrows = self.X_train.shape[0]  # nb of rows to train on

    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        if estimator == "Logic":
            model = LogisticRegression(max_iter=1000)
        # elif estimator == "RandomForest":
        #     model = RandomForestRegressor()
        #     self.model_params = {   'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
        #         'max_features': ['auto', 'sqrt']}
        #      'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        # else:
        #     model = Lasso()
        # estimator_params = self.kwargs.get("estimator_params", {})
        # model.set_params(**estimator_params)
        # print(model.__class__.__name__)
        return model


    def set_pipeline(self):
        """implement the feature pipelines"""
        pass

    def train(self, df):
        SCORE_COLS = ['score_1', 'score_2', 'score_3', 'score_4', 'score_5', 'overall']
        """iterate features, train and predict targets"""
        predictions = pd.DataFrame()
        pred_scores = {}
        for target in SCORE_COLS:
            # self.set_pipeline(): after set the pipelines
            self.model = self.get_estimator()
            y = df[target] # need feature packages
            self.model.fit(self.X, y)
            predictions[target] = self.model.predict(self.X)
            pred_scores[target] = self.model.score(self.X, y)
        return (predictions, pred_scores)


if __name__ == "__main__":
    # Get clean merged data
    # N = 1000 & add nrows=1000 to get_data() if needed
    submission, train, test = get_data()
    data = merge(submission, train, test)
    X_train = holdout(data, target)[0]
    y_train = holdout(data, target)[2]
    X_val = holdout(data, target)[1]
    y_val = holdout(data, target)[3]

    # Train and save model, locally and
    #t = Trainer(X=X_train, y=y_train, estimator="Linear")
    #t.train()
