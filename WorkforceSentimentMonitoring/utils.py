def lowercase (text):
    lowercased = text.lower()
    return lowercased

def remove_punctuation(text):
    import string
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    return text

def remove_numbers (text):
    """remove numbers"""
    words_only = ''.join([i for i in text if not i.isdigit()])
    return words_only

def remove_stopwords(text):
    """remove stopwords + tokenize"""
    from nltk.corpus import stopwords
    from nltk import word_tokenize
    stop_words = set(stopwords.words('english'))

    tokenized = word_tokenize(text)
    without_stopwords = [word for word in tokenized if not word in stop_words]
    return without_stopwords

def lemmatize(text):
    """Lemmatize text"""
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer() # Initiate lemmatizer
    lemmatized = [lemmatizer.lemmatize(word) for word in text] # Lemmatize
    lemmatized_string = " ".join(lemmatized)
    return lemmatized_string

def extract_negative(df):
    """return df with negative reviews and their labels"""
    # TODO
    pass

def extract_positive(df):
    """return df with positive reviews and their labels"""
    # TODO
    pass
