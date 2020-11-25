# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" Main lib for WorkforceSentimentMonitoring Project
"""

from os.path import split
import pandas as pd
import datetime

pd.set_option('display.width', 200)


# def clean_data(data):
#     """ clean data
#     """
#     # Remove columns starts with vote
#     cols = [x for x in data.columns if x.find('vote') >= 0]
#     data.drop(cols, axis=1, inplace=True)
#     # Remove special characteres from columns
#     data.loc[:, 'civility'] = data['civility'].replace('\.', '', regex=True)
#     # Calculate Age from day of birth
#     actual_year = datetime.datetime.now().year
#     data.loc[:, 'Year_Month'] = pd.to_datetime(data.birthdate)
#     data.loc[:, 'Age'] = actual_year - data['Year_Month'].dt.year
#     # Uppercase variable to avoid duplicates
#     data.loc[:, 'city'] = data['city'].str.upper()
#     # Take 2 first digits, 2700 -> 02700 so first two are region
#     data.loc[:, 'postal_code'] = data.postal_code.str.zfill(5).str[0:2]
#     # Remove columns with more than 50% of nans
#     cnans = data.shape[0] / 2
#     data = data.dropna(thresh=cnans, axis=1)
#     # Remove rows with more than 50% of nans
#     rnans = data.shape[1] / 2
#     data = data.dropna(thresh=rnans, axis=0)
#     # Discretize based on quantiles
#     data.loc[:, 'duration'] = pd.qcut(data['surveyduration'], 10)
#     # Discretize based on values
#     data.loc[:, 'Age'] = pd.cut(data['Age'], 10)
#     # Rename columns
#     data.rename(columns={'q1': 'Frequency'}, inplace=True)
#     # Transform type of columns
#     data.loc[:, 'Frequency'] = data['Frequency'].astype(int)
#     # Rename values in rows
#     drows = {1: 'Manytimes', 2: 'Onetimebyday', 3: '5/6timesforweek',
#              4: '4timesforweek', 5: '1/3timesforweek', 6: '1timeformonth',
#              7: '1/trimestre', 8: 'Less', 9: 'Never'}
#     data.loc[:, 'Frequency'] = data['Frequency'].map(drows)
#     return data

def get_df():
    path = os.path.join(os.getcwd(), '../raw_data')
    submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    test = pd.read_csv(os.path.join(path,'test.csv'))

    test = pd.merge(test, submission, on=['ID'])
    frames = [train, test]
    df = pd.concat(frames)
    df = df.rename(columns={'score_1':'work-balance', 'score_2':'culture-values', 'score_3':'career-opportunities', 'score_4':'comp-benefits', 'score_5':'senior-mgmt', 'score_6':'helpful-count'})
    df = df.drop(columns=['ID', 'Place', 'location', 'date', 'status', 'job_title', 'helpful-count'])
    # create a review column containing all text information
    text_columns = ['summary', 'positives', 'negatives', 'advice_to_mgmt']
    df['review'] = df[text_columns].astype('U').agg(' '.join, axis=1)
    # drop missing values
    categories = ['work-balance', 'culture-values', 'career-opportunities', 'comp-benefits', 'senior-mgmt', 'overall']
    df = df.dropna(axis=0, subset=categories)
    df = df.drop_duplicates()
    df[categories] = df[categories].astype('uint8')
    return df

if __name__ == '__main__':
    # For introspections purpose to quickly get this functions on ipython
    import WorkforceSentimentMonitoring
    folder_source, _ = split(WorkforceSentimentMonitoring.__file__)
    df = pd.read_csv('{}/data/data.csv.gz'.format(folder_source))
    get_df = get_df()
    print(' dataframe fetched')
