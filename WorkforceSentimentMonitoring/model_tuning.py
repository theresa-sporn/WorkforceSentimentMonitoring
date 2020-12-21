def set_pipe_dfidf_nb():
    vectorizer = ColumnTransformer([
            ('vectorizer' ,TfidfVectorizer(), 'review')
        ],
        remainder='drop')

    pipe = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', MultinomialNB())
        ])
    return pipe

def random_gridsearch_pipe(pipe, X, y, **kwargs):
    grid = dict(
        vectorizer__vectorizer__analyzer = ['char', 'word'],
        vectorizer__vectorizer__ngram_range = [(1,2), (1,3), (1,4), (1,5), (1,6), (1,7)],
        vectorizer__vectorizer__max_df = loguniform(0.7, 1.0),
        vectorizer__vectorizer__min_df = loguniform(0.001, 0.1),
        vectorizer__vectorizer__stop_words = [None, 'english'],
        vectorizer__vectorizer__norm = ['l1', 'l2'],
        classifier__alpha = loguniform(0.001, 1)
    )

    gridsearch = RandomizedSearchCV(pipe, grid, n_iter=10,
                                verbose=1, refit=True,
                                scoring='balanced_accuracy', n_jobs=-1)
    gridsearch.fit(X, y)

    return gridsearch.best_estimator_

def export_joblib(pipe, name):
    dirname = os.path.abspath('')
    filename = os.path.join(dirname, f'../joblib_files/{name}.joblib')
    joblib.dump(pipe, filename)


def iterative_gridsearch_pipe(pipe, X, y, predict=True, export_joblib=True):
    targets = y.columns
    estimators = []
    for target in tqdm(targets):
        best_estimator = random_gridsearch_pipe(pipe, X, y[target])
        estimators.append(best_estimator)
        if export_joblib:
            feature_name = f'{target}_nb'
            export_joblib(best_estimator, feature_name)
        if predict:
            X[feature_name] = best_estimator.predict(X)
            return estimators, X
        return estimators
