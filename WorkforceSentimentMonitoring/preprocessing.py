import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import contractions
import re

def lowercase(text):
    """lowercase"""
    text = [x.lower() for x in text]
    return text

def remove_punctuation(text):
    text = re.sub("[^\w\d\']+", ' ', text).strip()
    return text

def remove_numbers(text):
    text = re.sub("[\d]+", '', text)
    return text

def remove_stopwords(text):
    """remove stopwords + tokenize"""
    stop_words = set(stopwords.words('english'))
    tokenized = word_tokenize(text)
    without_stopwords = [word for word in tokenized if not word in stop_words]
    without_stopwords_string = " ".join(without_stopwords)
    return without_stopwords_string

def lemmatize(text):
    """Lemmatize text"""
    lemmatizer = WordNetLemmatizer() # Initiate lemmatizer
    lemmatized = [lemmatizer.lemmatize(word) for word in text.split(" ")] # Lemmatize
    lemmatized_string = " ".join(lemmatized)
    text = lemmatized_string
    return text

def tokenize(df):
    tokenized_text = word_tokenize(str(df))
    return tokenized_text


def expand_contractions(text):
    """Fix word contractions like <I'm> into <I am>."""
    return contractions.fix(text)


def preprocessing(text, to_lower, words_only, rm_stopwords):

    if type(text) is not str:
        return text
    text = expand_contractions(text)
    if to_lower:
        text = text.lower()
    if words_only:
        text = remove_numbers(text)
        text = remove_punctuation(text)
    if rm_stopwords:
        text = remove_stopwords(text)
    text = re.sub("\d+", ' ', text)
    text = lemmatize(text)
    return text


