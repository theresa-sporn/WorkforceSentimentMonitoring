from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from textblob import TextBlob

# Create a function to get the subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# Create a function to get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity


def get_subjectivity_polarity_columns(df):
    text_columns = df.select_dtypes('object').columns
    for feature in text_columns:
        df[f'subjectivity_{feature}'] = df[feature].apply(getSubjectivity)
        df[f'polarity_{feature}'] = df[feature].apply(getPolarity)
    return df

def add_multinomial_nb_prediction_feature(df, y):
    """vectorize and predict with Naive Bayes"""

    feature_cols = ['summary', 'positives', 'negatives', 'advice_to_mgmt', 'review']

    for score in y.columns:

        for feature in feature_cols:
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df[feature])
            target = y[score]
            model = MultinomialNB()
            model.fit(X, target)
            df[f'{feature}_{score}_nb'] = model.predict(X)

    return df