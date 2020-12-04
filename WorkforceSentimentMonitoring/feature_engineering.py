from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
import joblib

pd.options.mode.chained_assignment = None


FEATURE_COLS = ["summary", "positives", "negatives", "advice_to_mgmt", "review"]
SCORE_COLS = [
    "work-balance",
    "culture-values",
    "career-opportunities",
    "comp-benefits",
    "senior-mgmt",
    "overall",
]


def get_lengths(df):
    '''returns a df with columns with the length of the reviews'''
    func = lambda x: len(x) if type(x) == str else 0
    for feature in tqdm(FEATURE_COLS):
        df[f'{feature}_length'] = df[feature].apply(func)
    return df


def getSubjectivity(text):
    if pd.isnull(text): return text
    return TextBlob(text).sentiment.subjectivity


def getPolarity(text):
    if pd.isnull(text): return text
    return TextBlob(text).sentiment.polarity


def get_subjectivity_polarity_columns(df):
    text_columns = df.select_dtypes("object").columns
    for feature in tqdm(text_columns):
        df[f"subjectivity_{feature}"] = df[feature].apply(getSubjectivity)
        df[f"polarity_{feature}"] = df[feature].apply(getPolarity)
    return df

def export_joblib(estimator, name):
    dirname = os.path.abspath('')
    filename = os.path.join(dirname, f'../joblib_files/{name}.joblib')
    joblib.dump(estimator, filename)


def extract_NB_predictions(X, y, targets):
    for target in tqdm(targets):
        vectorizer = ColumnTransformer([
            ('vectorizer' ,TfidfVectorizer(), 'review')
        ], remainder='drop')

        pipe = make_pipeline(
            (vectorizer),
            (MultinomialNB())
        )
        pipe.fit(X, y[target])
        feature_name = f'{target}_nb'
        export_joblib(pipe, feature_name)
        X[feature_name] = pipe.predict(X)
    return X


def get_linear_regression_cols(X_train, X_test, y_train):
    """linear regression with just the predictions for each model"""
    # iterate over the targets and instantiate, fit one model for each target
    for col in tqdm(SCORE_COLS):
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
    for target in tqdm(SCORE_COLS):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train[target])
        prediction_scores_dict[target] = model.score(X_test, y_test[target])

    return model, prediction_scores_dict


def create_wordcount_vector(corpus):
    """Vectorize corpus. Corpus is a pd.Series with texts"""
    vectorizer = CountVectorizer(stop_words='english', strip_accents='ascii')
    X_vectorized = vectorizer.fit_transform(corpus)
    X_vectorized = X_vectorized.toarray()
    columns = vectorizer.get_feature_names()
    X_vectorized = pd.DataFrame(X_vectorized, columns=columns)
    return X_vectorized


def create_tfidf_vector(corpus):
    """Vectorize corpus. Corpus is a pd.Series with texts"""
    vectorizer = TfidfVectorizer(strip_accents='ascii')
    X_vectorized = vectorizer.fit_transform(corpus)
    X_vectorized = X_vectorized.toarray()
    columns = vectorizer.get_feature_names()
    X_vectorized = pd.DataFrame(X_vectorized, columns=columns)
    return X_vectorized


def create_emotion_dictionary(lexicon):
    """Create dict with word : emo_array pairs"""
    # create pivot table to better extract the word : array pairs
    table = pd.pivot_table(lexicon, values='emotion-intensity-score',
                           index='word', columns='emotion', fill_value=0)
    # create dictionary
    emo_scores_dict = {word : value for word , value in zip(table.index, table.values)}
    return emo_scores_dict


def simplify_emotion_dict_and_wordcount(emo_scores_dict, word_count_vec):
    """Deletes the emotion keys in the dictionary that aren't present in the dataset"""
    # Create set intersection with words appearing in both the dic and the vector
    columns_intersection = set(word_count_vec.columns).intersection(set(emo_scores_dict.keys()))
    # Drop unnecessary word columns in word_count_vec
    word_count_vec = word_count_vec[columns_intersection]
    # Drop innecessary entries in emo_scores_dict
    keys_to_drop = set(emo_scores_dict.keys()).difference(columns_intersection)
    for key in keys_to_drop:
        emo_scores_dict.pop(key)
    return emo_scores_dict, word_count_vec


def get_emotion_score(X, lexicon):
    """Extract emotion scores"""
    # create pivot table to better extract the word : array pairs
    table = pd.pivot_table(lexicon, values='emotion-intensity-score',
                           index='word', columns='emotion', fill_value=0)

    X_vectorized = create_tfidf_vector(X['review'])
    emo_scores_dict = create_emotion_dictionary(lexicon)
    emo_scores_dict, X_vectorized = simplify_emotion_dict_and_wordcount(emo_scores_dict,
                                                                        X_vectorized)
    X['length'] = X.review.str.split(' ').apply(len)
    emotions = lexicon.emotion.unique()

    # Create new empty columns for emotion_scores
    for emo in emotions:
        X[f'{emo}_score'] = np.nan
    # iterate through every row
    for i in tqdm(range(len(X))):
        # select columns containing words in the word count vector
        col_selector = X_vectorized.loc[i] > 0
        review = X_vectorized.loc[i, col_selector]
        # create an empty np.array with 8 spaces to add the results to
        emo_score = np.zeros(8)
        # iterate over the words contained in the review
        for j in range(len(review)):
            # select the word (string)
            word = review.index[j]
            # select the count (int)
            word_count = review[j]
            # compute emo_score by multiplying the array from the dict with the
            # word count
            emo_array = emo_scores_dict[word] * word_count
            # add emo_array to emo_score array
            emo_score += emo_array
        # iterate over the emotion columns to append the corresponding value
        for idx, emo in enumerate(emotions):
            X[f'{emo}_score'][i] = emo_score[idx]

    return X

def get_mnb_features(X):

    path = os.path.split(os.path.abspath(__file__))[0]
    model_path = os.path.join(path, "../joblib_files")

    # list of tuple(col_names, path_to_model)
    list_col_model = [('work-balance_nb', 'work-balance_nb.joblib'),
                      ('culture-values_nb', 'culture-values_nb.joblib'),
                      ('career-opportunities_nb', 'career-opportunities_nb.joblib'),
                      ('comp-benefits_nb', 'comp-benefits_nb.joblib'),
                      ('senior-mgmt_nb', 'senior-mgmt_nb.joblib'),
                      ('overall_nb', 'overall_nb.joblib')]
    # iterate the tuple
    for col_model in list_col_model:
        col = col_model[0]
        model_name = col_model[1]
        model = joblib.load(os.path.join(model_path, model_name))
        X[col] = model.predict(X)#.iloc[:,:1])
    return X

def get_clf_scores(X):

    path = os.path.split(os.path.abspath(__file__))[0]
    model_path = os.path.join(path, "../joblib_files")

    # list of tuple(col_names, path_to_model)
    list_col_model = [('work-life-balance', 'work-balance_clf.joblib'),
                      ('culture-values', 'culture-values_clf.joblib'),
                      ('career-opportunities', 'career-opportunities_clf.joblib'),
                      ('company-benefits', 'comp-benefits_clf.joblib'),
                      ('senior-management', 'senior-mgmt_clf.joblib'),
                      ('overall', 'overall_clf.joblib')]
    # iterate the tuple
    for col_model in list_col_model:
        col = col_model[0]
        model_name = col_model[1]
        model = joblib.load(os.path.join(model_path, model_name))
        X[col] = model.predict(X.iloc[:,:17])
    return X[X.columns[-6:]]
