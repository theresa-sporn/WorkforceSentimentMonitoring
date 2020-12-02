import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

def lowercase(text):
	"""lowercase"""
	lowercased = text.lower()
	return lowercased

def remove_punctuation(text):
	"""remove punctuation"""
	for punctuation in string.punctuation:
		text = text.replace(punctuation, ' ')
	return text

def remove_numbers(text):
	"""remove numbers"""
	words_only = ''.join([i for i in text if not i.isdigit()])
	return words_only

def remove_stopwords(text):
	"""remove stopwords + tokenize"""
	stop_words = set(stopwords.words('english'))
	tokenized = word_tokenize(text)
<<<<<<< HEAD
	without_stopwords = " ".join([word for word in tokenized if not word in stop_words])
	return without_stopwords
=======
	without_stopwords = [word for word in tokenized if not word in stop_words]
	without_stopwords_string = " ".join(without_stopwords)
	return without_stopwords_string
>>>>>>> d0b09363ff13ef202246a0bb60c666301fac24ac

def lemmatize(text):
	"""Lemmatize text"""
	lemmatizer = WordNetLemmatizer() # Initiate lemmatizer
	lemmatized = [lemmatizer.lemmatize(word) for word in text.split(" ")] # Lemmatize
	lemmatized_string = " ".join(lemmatized)
	return lemmatized_string

def tokenize(df):
    tokenized_text = word_tokenize(str(df))
    return tokenized_text

def preprocessing(text, to_lower, words_only, rm_stopwords):

	if type(text) is not str:
		return text

	text = text.strip()
	if to_lower:
		text = text.lower()
	if words_only:
		text = remove_numbers(text)
		text = remove_punctuation(text)
	if rm_stopwords:
		text = remove_stopwords(text)

	text = lemmatize(text)
	return text


