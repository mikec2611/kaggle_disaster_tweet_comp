import kaggle
import nltk
import pandas as pd
import numpy as np
import inflect
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer, PorterStemmer
import re
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from scipy.sparse import csr_matrix, hstack

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# -steps
# get data 
# split data
# describe data
# clean text
# tokenize text
# remove stopwords
# stem text
# vectorize text

def clean_text(text, i_eng):

	text.fillna('', inplace=True) # replace nan

	if isinstance(text, pd.Series):
		text = text.apply(lambda x:x.lower()) # lowercase
		text = text.apply(lambda x:re.sub('https?://\S+|www\.\S+', '', x) ) # remove hyperlinks
		text = text.apply(lambda x:re.sub('\[.*?\]', '', x) ) # remove brackers
		text = text.apply(lambda x:re.sub('<.*?>+', '', x) )  # remove <>
		text = text.apply(lambda x:re.sub('[%s]' % re.escape(string.punctuation), '', x) )# remove punctuation
		text = text.apply(lambda x:re.sub('\n' , '', x) ) # remove newline
		text = text.apply(lambda row: re.sub(r'(?<!\S)\d+(?!\S)', lambda x: i_eng.number_to_words(x.group()), row)) # convert numbers to words
		# text = text.apply(lambda x:re.sub(' +', ' ', x)) # remove spaces
		
	else:
		text = text.applymap(lambda x:x.lower()) # lowercase
		text = text.applymap(lambda x:re.sub('https?://\S+|www\.\S+', '', x) ) # remove hyperlinks
		text = text.applymap(lambda x:re.sub('\[.*?\]', '', x) ) # remove brackers
		text = text.applymap(lambda x:re.sub('<.*?>+', '', x) )  # remove <>
		text = text.applymap(lambda x:re.sub('[%s]' % re.escape(string.punctuation), '', x) ) # remove punctuation
		text = text.applymap(lambda x:re.sub('\n' , '', x) ) # remove newline
		text = text.applymap(lambda row: re.sub(r'(?<!\S)\d+(?!\S)', lambda x: i_eng.number_to_words(x.group()), row)) # convert numbers to words
		# text = text.applymap(lambda x:re.sub(' +', ' ', x)) # remove spaces

	return text

def prep_data(data, stop_words, stemmer, lemmatizer):
	if isinstance(data, pd.Series):
		data = data.apply(lambda x:word_tokenize(x)) # tokenize 
		data = data.apply(lambda x: [w for w in x if not w in stop_words]) # stopwords
		# data = data.apply(lambda x:" ".join(stemmer.stem(token) for token in x)) # stem
		# data = data.apply(lambda x:lemmatizer.lemmatize(x)) # lemmatize
	else:
		data = data.applymap(lambda x:word_tokenize(x)) # tokenize 
		data = data.applymap(lambda x: [w for w in x if not w in stop_words]) # stopwords
		# data = data.applymap(lambda x:" ".join(stemmer.stem(token) for token in x)) # stem
		# data = data.applymap(lambda x:lemmatizer.lemmatize(x)) # lemmatize

	data = data.apply(lambda x: ''.join(i+' ' for i in x)) # back to string

	return data

def fit_models(X_train, X_test, y_train, y_test):
	# models = [LogisticRegression(), MultinomialNB(), RandomForestClassifier(), KNeighborsClassifier()]
	models = [LogisticRegression(), MultinomialNB()]
	scores = []
	for model in models:
		model.fit(X_train, y_train)
		model_score = cross_val_score(model, X_test, y_test, cv=5, scoring="f1")
		scores.append((model, np.mean(model_score)))

	scores.sort(key=lambda x: x[1], reverse=True)
	print(scores)

	return scores[0]



def run_process():
	# get data
	raw_data_train = pd.read_csv('train.csv')
	raw_data_test = pd.read_csv('test.csv')

	# settings for data prep
	i_eng = inflect.engine()
	stop_words = set(stopwords.words('english'))
	stemmer = PorterStemmer()
	lemmatizer = WordNetLemmatizer()
	
	# data_train = raw_data_train[['keyword', 'location', 'text']]
	# data_test = raw_data_train[['keyword', 'location', 'text']]
	data_train = raw_data_train.text
	data_test = raw_data_test.text

	data_train = clean_text(data_train, i_eng)
	data_test = clean_text(data_test, i_eng)

	data_train = prep_data(data_train, stop_words, stemmer, lemmatizer)
	data_test = prep_data(data_test, stop_words, stemmer, lemmatizer)

	# select data for training
	X = data_train
	y = raw_data_train.target
	test_pred = data_test

	# split data
	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

	# vectorize 
	vectorizer = CountVectorizer()
	X_train = vectorizer.fit_transform(X_train)
	X_test = vectorizer.transform(X_test)
	test_pred = vectorizer.transform(test_pred)

	# X_train = hstack([vectorizer.fit_transform(X_train.text),vectorizer.fit_transform(X_train.location), vectorizer.fit_transform(X_train.keyword)], 'csr')
	# X_test = hstack([vectorizer.transform(X_test.text),vectorizer.transform(X_test.location), vectorizer.transform(X_test.keyword)], 'csr')
	# test_pred = hstack([vectorizer.transform(test_pred.text),vectorizer.transform(test_pred.location), vectorizer.transform(test_pred.keyword)], 'csr')
	# X = hstack([vectorizer.fit_transform(X.text),vectorizer.fit_transform(X.location), vectorizer.fit_transform(X.keyword)], 'csr')
	# X_test = hstack([vectorizer.fit_transform(X_test.text),vectorizer.fit_transform(X_test.location), vectorizer.fit_transform(X_test.keyword)], 'csr')
	# test_pred = hstack([vectorizer.transform(test_pred.text),vectorizer.transform(test_pred.location), vectorizer.transform(test_pred.keyword)], 'csr')

	best_model, score = fit_models(X_train, X_test, y_train, y_test)

	# save model
	pickle.dump(best_model, open('model', 'wb'))
	loaded_model = pickle.load(open('model', 'rb'))

	# predictions
	predictions = loaded_model.predict(X_test)

	# prediction scores
	print(loaded_model, confusion_matrix(y_test,predictions))
	print(loaded_model, classification_report(y_test,predictions))
	print(loaded_model, metrics.accuracy_score(y_test, predictions)*100)

	# save predictions
	results = np.array(list(zip(raw_data_test.id,predictions)))
	results = pd.DataFrame(results, columns=['id', 'target'])
	results.to_csv('nb_results.csv', index = False)

run_process()