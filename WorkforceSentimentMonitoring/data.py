# merging and cleaning of dataset

import pandas as pd

def get_data():

    path_submission = os.path.dirname(os.path.abspath(WorkforceSentimentMonitoring.__file__)) + '/raw_data/sample_submission.csv'
    path_train = os.path.dirname(os.path.abspath(WorkforceSentimentMonitoring.__file__)) + '/raw_data/train.csv'
    path_tes = os.path.dirname(os.path.abspath(WorkforceSentimentMonitoring.__file__)) + '/raw_data/test.csv'

    submission = pd.read_csv(path_submission)
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)

    return submission, train, test


def merge(submission, train, test):

    test = pd.merge(test, submission, on=['ID'])
    frames = [train, test]
    df = pd.concat(frames)
    df = df.rename(columns={'score_1':'work-balance', 'score_2':'culture-values', 'score_3':'carreer-opportunities', 'score_4':'comp-benefit', 'score_5':'senior-mangemnet', 'score_6':'helpful-count'})

    return df

def holdout(df):

    y = df["overall"]
    X = df.drop("overall", axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

    return (X_train, X_val, y_train, y_val)


if __name__ == '__main__':
    df = get_data()
    print(df)
