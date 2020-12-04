import warnings
import joblib
import os

import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin

from WorkforceSentimentMonitoring.data import get_data, merge, holdout, get_prepaired_data, drop_wrong_language
from WorkforceSentimentMonitoring.feature_engineering import get_mnb_features, get_clf_scores
from WorkforceSentimentMonitoring.preprocessing import preprocessing
from WorkforceSentimentMonitoring.encoders import Preprocessor, CustomMinMaxScaler, FeatureEngineer
from test_data.backup_test_data import get_test_data
from io import BytesIO
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

# FEATURE_COLS = ['positives', 'negatives', 'review']
# SCORE_COLS = ['work-balance', 'culture-values', 'career-opportunities', 'comp-benefits', 'senior-mgmt', 'overall']


class MultiNBFeaturesExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.text_columns = X.select_dtypes('object').columns
        return self

    def transform(self, X, y=None):
        return get_mnb_features(X[self.text_columns])


class Classifier(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.X = self.kwargs.get('drop_wrong_language', True)


    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        return self

    def transform(self, X, y=None):
        self.X = drop_wrong_language(X, 'review')
        return get_clf_scores(X=self.X.drop(columns='review'))

# def get_mnb_features(df):

#     path = os.path.split(os.path.abspath(__file__))[0]
#     model_path = os.path.join(path, "../joblib_files")
#     #model_path = "https://github.com/theresa-sporn/WorkforceSentimentMonitoring/blob/master/joblib_files"

#     model_link = "https://github.com/theresa-sporn/WorkforceSentimentMonitoring/blob/master/joblib_files"
#     model_file = BytesIO(requests.get(model_link).content)
#     # list of tuple(col_names, path_to_model)
#     list_col_model = [('work-balance_nb', 'work-balance_nb.joblib'),
#                       ('culture-values_nb', 'culture-values_nb.joblib'),
#                       ('career-opportunities_nb', 'career-opportunities_nb.joblib'),
#                       ('comp-benefits_nb', 'comp-benefits_nb.joblib'),
#                       ('senior-mgmt_nb', 'senior-mgmt_nb.joblib'),
#                       ('overall_nb', 'overall_nb.joblib')]
#     # iterate the tuple
#     for col_model in list_col_model:
#         col = col_model[0]
#         model_name = col_model[1]
#         model = joblib.load(os.path.join(model_file, model_name))
#         df[col] = model.predict(df)
#     return df





# class Trainer(object):
#     SCORE_COLS = ['work-balance', 'culture-values', 'career-opportunities', 'comp-benefits', 'senior-mgmt', 'overall']

#     def __init__(self, data, **kwargs):
#         self.kwargs = kwargs
#         self.data = get_test_data()
#         self.model = self.kwargs.get('saved_model', 'model.joblib')

#     def get_estimator(self):
#         estimator = self.kwargs.get("estimator", self.ESTIMATOR)
#         if estimator == "Logic":
#             model = LogisticRegression(max_iter=10000)
#         elif estimator == "AdaBoost":
#             model = AdaBoostClassifier(
#                 #base_estimator=RandomForestClassifier(class_weight='balanced', max_depth=5),
#                 #n_estimators=100,
#                 #learning_rate=0.1
#                 )
#         elif estimator == "GradientBoost":
#             model = GradientBoostingClassifier()
#         elif estimator == "XGB":
#             model = XGBClassifier()
#         elif estimator == "RandomForest":
#             model = RandomForestClassifier(
#                 class_weight='balanced',
#                 max_features='auto',
#                 max_depth=5,
#                 n_estimators=1000,
#                 )
#             self.model_params = {
#             #'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
#             'max_features': ['auto', 'sqrt'],
#             #'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]
#             }
#         else:
#             model = VotingClassifier()
#         # estimator_params = self.kwargs.get("estimator_params", {})
#         # model.set_params(**estimator_params)
#         print(model.__class__.__name__)
#         return model


#     def set_pipeline(self):
#         """implement the feature pipelines"""

#         pass

#     def add_grid_search(self):
#         """"
#         Apply Gridsearch on self.params defined in get_estimator
#         {'rgs__n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
#           'rgs__max_features' : ['auto', 'sqrt'],
#           'rgs__max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
#         """
#         # Here to apply ramdom search to pipeline, need to follow naming "rgs__paramname"
#         params = {"rgs__" + k: v for k, v in self.model_params.items()}
#         self.pipeline = RandomizedSearchCV(
#             estimator=self.pipeline,
#             param_distributions=params,
#             n_iter=10,
#             cv=2,
#             verbose=1,
#             random_state=42,
#             n_jobs=-1
#             )
#         #pre_dispatch=None)

#     @simple_time_tracker
#     def model_train(self, y_train, y_test): # old version
#         tic = time.time()
#         #SCORE_COLS = ['work-balance', 'culture-values', 'career-opportunities', 'comp-benefits', 'senior-mgmt', 'overall']
#         """iterate features, train and predict targets"""
#         predictions = pd.DataFrame()
#         pred_scores = {}
#         for target in SCORE_COLS:
#             # self.set_pipeline(): after set the pipelines
#             if self.gridsearch:
#                 self.add_grid_search()
#             self.model = self.get_estimator() # need feature packages
#             self.model.fit(self.X_train, y_train[target])
#             predictions[target] = self.model.predict(self.X_test)
#             pred_scores[target] = self.model.score(self.X_test, y_test[target])
#             print(classification_report(self.model.predict(self.X_test), y_test[target]))
#         return (predictions, pred_scores)


#     def train(self):
#         '''new version'''
#         prediction_scores_dict = {}
#         for target in tqdm(SCORE_COLS):
#             # self.set_pipeline(): after set the pipelines
#             if self.gridsearch:
#                 self.add_grid_search()
#             self.model = self.get_estimator()
#             self.model.fit(self.X_train, y_train[target])
#             prediction_scores_dict[target] = self.model.score(self.X_test, y_test[target])

#         return self.model, prediction_scores_dict


#     def load_model(self):
#         clf = joblib.load('clf_model.joblib')
#         print('model loaded')
#         y_pred = clf.predict(X)


#     def save_model(self):
#         """Save the model into a .joblib and upload it somewhere or save locally"""
#         joblib.dump(self.pipeline, 'model.joblib')
#         print("model.joblib saved locally")

#         if self.upload:
#             storage_upload(model_version=MODEL_VERSION)


# if __name__ == "__main__":
#     # Get clean merged data
#     # N = 1000 & add nrows=1000 to get_data() if needed
#     submission, train, test = get_data()
#     data = merge(submission, train, test)
#     X_train = holdout(data, target)[0]
#     y_train = holdout(data, target)[2]
#     X_val = holdout(data, target)[1]
#     y_val = holdout(data, target)[3]

#     # Train and save model, locally and
#     #t = Trainer(X=X_train, y=y_train, estimator="Linear")
#     #t.train()
