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
	without_stopwords = [word for word in tokenized if not word in stop_words]
	return without_stopwords

def lemmatize(text):
	"""Lemmatize text"""
	lemmatizer = WordNetLemmatizer() # Initiate lemmatizer
	lemmatized = [lemmatizer.lemmatize(word) for word in text] # Lemmatize
	lemmatized_string = " ".join(lemmatized)
	return lemmatized_string

def preprocessing(text, to_lower, words_only, rm_stopwords):

	if pd.isnull(text): return text

	if to_lower:
		text = text.lower()
	elif words_only:
		text = remove_numbers(text)
		text = remove_punctuation(text)
	elif rm_stopwords:
		text = remove_stopwords(text)
	else:
		text = lemmatize(text)
	return text
