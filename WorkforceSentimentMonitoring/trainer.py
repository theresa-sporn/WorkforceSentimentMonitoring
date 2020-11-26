import warning

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


class Trainer(object):
    ESTIMATOR = "MultinomialNB"

    def __init__(self, X, y, **kwargs):
        #self.pipeline = None
        self.kwargs = kwargs
        self.split = self.kwargs.get("split", True)  # cf doc above
        self.X_train = X
        self.y_train = y
        # self.nrows = self.X_train.shape[0]  # nb of rows to train on

    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        if estimator == "MultinomialNB":
            model = MultinomialNB()
        elif estimator == "Linear":
            model = LinearRegression()
        # elif estimator == "RandomForest":
        #     model = RandomForestRegressor()
        #     self.model_params = {   'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
        #         'max_features': ['auto', 'sqrt']}
        #      'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        else:
            model = LogisticRegression()
        estimator_params = self.kwargs.get("estimator_params", {})
        model.set_params(**estimator_params)
        print(model.__class__.__name__)
        return model

    def set_pipeline(self):

        # feature_engineer = make_pipeline(get_subjectivity(column='text'),
        #                               TextBlob())

        # pipe_polarity = make_pipeline(get_polarity(column='text'), TextBlob())

        # features_encoder = ColumnTransformer([
        #     ('text', DistanceTransformer(**DIST_ARGS), list(DIST_ARGS.values())),
        #     ('', time_features, ['']),
        #     )
        # ])

        self.pipeline = Pipeline(steps=[
            ('tfidf', TfidfVectorizer()),
            ('rgs', self.get_estimator())])


    def train(self):
        tic = time.time()
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self, X_test, y_test):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        #rmse = compute_rmse(y_pred, y_test)
        return


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
    t = Trainer(X=X_train, y=y_train, estimator="RandomForest")
    t.train()
