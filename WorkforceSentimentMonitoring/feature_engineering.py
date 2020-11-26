from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB


FEATURE_COLS = ["summary", "positives", "negatives", "advice_to_mgmt", "review"]
SCORE_COLS = [
    "work-balance",
    "culture-values",
    "career-opportunities",
    "comp-benefits",
    "senior-mgmt",
    "overall",
]


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def getPolarity(text):
    return TextBlob(text).sentiment.polarity


def get_subjectivity_polarity_columns(df):
    text_columns = df.select_dtypes("object").columns
    for feature in tqdm(text_columns):
        df[f"subjectivity_{feature}"] = df[feature].apply(getSubjectivity)
        df[f"polarity_{feature}"] = df[feature].apply(getPolarity)
    return df


def add_multinomial_nb_prediction_feature(df, y):
    """vectorize and predict with Naive Bayes"""

    feature_cols = FEATURE_COLS

    for score in y.columns:

        for feature in feature_cols:
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df[feature])
            target = y[score]
            model = MultinomialNB()
            model.fit(X, target)
            df[f"{feature}_{score}_nb"] = model.predict(X)

    return df


def get_linear_regression_cols(X_train, X_test, y_train):
    """linear regression with just the predictions for each model"""
    # iterate over the targets and instantiate, fit one model for each target
    for col in tqdm(score_cols):
        # instantiate model
        model = LinearRegression()
        # fit model with train set
        model.fit(X_train, y_train[col])
        # predict and add values to new column (train set)
        X_train[f"{col}_regression"] = model.predict(X_train)
        # predict and add values to new column (test set), is this correct??
        # should I instantiate and fit a new model??
        X_test[f"{col}_regression"] = model.predict(X_test)

    return X_train, X_test


def minmax_when_needed(X_train, X_test):
    """Select columns with values that need scaling – outside of range(-1, 1)
    and apply a MinMax scaler"""
    # select columns the columns that need scaling
    columns_to_scale = [
        col
        for col in X_train.select_dtypes(exclude="object").columns
        if X_train[col].max() > 1 or X_train[col].min() < -1
    ]
    # instantiate scaler
    scaler = MinMaxScaler()
    # fit & transform train set, reassign
    X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    # transform test set, reassign
    X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    return X_train, X_test


def train_logReg(X_train, y_train, X_test, y_test, score_cols=SCORE_COLS):
    """Trains the a Logistic Regression model for every target class and returns
    it with a dictionary containing the scores of the validation with the test set"""
    prediction_scores_dict = {}
    for target in tqdm(score_cols):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train[target])
        prediction_scores[target] = model.score(X_test, y_test[target])

    return model, prediction_scores_dict
