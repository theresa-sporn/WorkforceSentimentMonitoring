import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from WorkforceSentimentMonitoring.data import get_prepaired_data
from WorkforceSentimentMonitoring.feature_engineering import get_lengths, getSubjectivity, getPolarity
from WorkforceSentimentMonitoring.preprocessing import preprocessing
from textblob import TextBlob
from tqdm import tqdm




class Preprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.to_lower = self.kwargs.get('to_lower', True)
        self.rm_stopwords = self.kwargs.get('rm_stopwords', False)
        self.words_only = self.kwargs.get('words_only', True)

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.text_columns = X.select_dtypes('object').columns
        self.preprocessing_params = dict(to_lower = self.to_lower,
                                         rm_stopwords = self.rm_stopwords,
                                         words_only = self.words_only)
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X[self.text_columns] = X[self.text_columns].applymap(lambda text: text if pd.isnull(text)
                                                             else preprocessing(text, **self.preprocessing_params))
        return X


class CustomMinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None, X_test=None):
        assert isinstance(X, pd.DataFrame)
        self.columns_to_scale = [col for col in X.select_dtypes(exclude='object').columns
                                 if X[col].max() > 1 or X[col].min() < -1]
        self.scaler.fit(X[self.columns_to_scale])
        return self

    def transform(self, X, y=None, X_test=None):
        assert isinstance(X, pd.DataFrame)
        X[self.columns_to_scale] = self.scaler.transform(X[self.columns_to_scale])
        if X_test:
            assert isinstance(X_test, pd.DataFrame)
            X_test[self.columns_to_scale] = self.scaler.transform(X_test[self.columns_to_scale])
            return X, X_test
        else:
            return X


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.polarity = self.kwargs.get('polarity', True)
        self.subjectivity = self.kwargs.get('subjectivity', True)
        self.length = self.kwargs.get('length', True)

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.text_columns = X.select_dtypes('object').columns
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        if self.length: X = self.get_length(X)
        if self.subjectivity or self.polarity:
            X = self.get_subjectivity_polarity_columns(X)
        return X

    def get_length(self, df):
        '''returns a df with columns with the length of the reviews'''
        assert isinstance(df, pd.DataFrame)
        func = lambda x: len(x) if type(x) is str else 0
        for feature in (self.text_columns):
            df[f'{feature}_length'] = df[feature].apply(func)
        return df

    def getSubjectivity(self, text):
        if pd.isnull(text): return text
        return TextBlob(text).sentiment.subjectivity

    def getPolarity(self, text):
        if pd.isnull(text): return text
        return TextBlob(text).sentiment.polarity

    def get_subjectivity_polarity_columns(self, df):
        assert isinstance(df, pd.DataFrame)
        for feature in tqdm(self.text_columns):
            if self.subjectivity:
                df[f"subjectivity_{feature}"] = df[feature].apply(self.getSubjectivity)
            if self.length:
                df[f"polarity_{feature}"] = df[feature].apply(self.getPolarity)
        return df


class PredictionFeaturesExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        self.text_columns = X.select_dtypes('object').columns
        self.score_cols = y.columns

    def transform(self, X, y):
        for score in score_cols:
            y_ = y[col]
            for feature in text_columns:
                model = MultinomialNB()
                X_ = vectorize(X[feature])
                model.fit(X_, y_)
                X[f'{feature}_{score}_nb'] = model.predict(X_, y_)
        return X

    def vectorize(self, feature_col):
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,7))
        return vectorizer.fit_transform(feature_col)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_prepaired_data()

    preprocessor = Preprocessor(to_lower=True)
    X_train = preprocessor.fit_transform(X_train)

    engineer = FeatureEngineer()
    X_train = engineer.fit_transform(X_train)

    scaler = CustomMinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    print(str(X_train.head(1)['review']))
    for col in X_train.columns:
        print(col)

