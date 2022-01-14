import kaggle
import nltk
import pandas as pd
import numpy as np
import inflect

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer, PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report
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


# get data
raw_data_train = pd.read_csv('train.csv')
raw_data_test = pd.read_csv('test.csv')

# view data
# print(raw_data_train.describe())
# print(raw_data_test.describe())
# print(raw_data_train.columns)
# print(raw_data_train.head())

data_train = raw_data_train[['keyword', 'location', 'text']]
data_test = raw_data_train[['keyword', 'location', 'text']]

data_train.fillna('', inplace=True)
data_test.fillna('', inplace=True)

print(data_train.head())

# clean text
# lowering the text
data_train = data_train.applymap(lambda x:x.lower())
data_test = data_test.applymap(lambda x:x.lower())
#removing square brackets
data_train = data_train.applymap(lambda x:re.sub('\[.*?\]', '', x) )
data_test = data_test.applymap(lambda x:re.sub('\[.*?\]', '', x) )
data_train = data_train.applymap(lambda x:re.sub('<.*?>+', '', x) )
data_test = data_test.applymap(lambda x:re.sub('<.*?>+', '', x) )
#removing hyperlink
data_train = data_train.applymap(lambda x:re.sub('https?://\S+|www\.\S+', '', x) )
data_test = data_test.applymap(lambda x:re.sub('https?://\S+|www\.\S+', '', x) )
#removing puncuation
data_train = data_train.applymap(lambda x:re.sub('[%s]' % re.escape(string.punctuation), '', x) )
data_test = data_test.applymap(lambda x:re.sub('[%s]' % re.escape(string.punctuation), '', x) )
data_train = data_train.applymap(lambda x:re.sub('\n' , '', x) )
data_test = data_test.applymap(lambda x:re.sub('\n', '', x) )
#remove words containing numbers
# data_train = data_train.apply(lambda x:re.sub('\w*\d\w*' , '', x))
# data_test = data_test.apply(lambda x:re.sub('\w*\d\w*', '', x) )
#convert numbers to words
p = inflect.engine()
data_train = data_train.applymap(lambda row: re.sub(r'(?<!\S)\d+(?!\S)', lambda x: p.number_to_words(x.group()), row))
data_test = data_test.applymap(lambda row: re.sub(r'(?<!\S)\d+(?!\S)', lambda x: p.number_to_words(x.group()), row))

# tokenize corpora
data_train = data_train.applymap(lambda x:word_tokenize(x))
data_test = data_test.applymap(lambda x:word_tokenize(x))

# print('token | ', data_train.head())

# remove stopwords
stop_words = set(stopwords.words('english'))
data_train = data_train.applymap(lambda x: [w for w in x if not w in stop_words])
data_train = data_test.applymap(lambda x: [w for w in x if not w in stop_words])

# print('remove stopwords | ', data_train.head())

# # stem - Snowball performed better than Porter
# stemmer = SnowballStemmer('english')
# data_train = data_train.applymap(lambda x:" ".join(stemmer.stem(token) for token in x))
# data_test = data_test.applymap(lambda x:" ".join(stemmer.stem(token) for token in x))

# # detokenize and lemmatize
lemmatizer = WordNetLemmatizer()
data_train = data_train.applymap(lambda x:TreebankWordDetokenizer().detokenize(x))
data_test = data_test.applymap(lambda x:TreebankWordDetokenizer().detokenize(x))
data_train = data_train.applymap(lambda x:lemmatizer.lemmatize(x))
data_test = data_test.applymap(lambda x:lemmatizer.lemmatize(x))

# print('stemmed | ', data_train.head())


X = data_train
y = raw_data_train.target
test_pred = data_test

print(X.head())

# vectorize 
count_vectorizer = CountVectorizer()
X = hstack([count_vectorizer.fit_transform(X.text),count_vectorizer.fit_transform(X.location), count_vectorizer.fit_transform(X.keyword)], 'csr')
test_pred = count_vectorizer.transform(test_pred)


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# MultinomialNB model
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_predictions = nb.predict(X_test)

print('NB',confusion_matrix(y_test,nb_predictions))
print('NB',classification_report(y_test,nb_predictions))
print('NB',metrics.accuracy_score(y_test, nb_predictions)*100)


# # GaussianNB model
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# gnb_predictions = gnb.predict(X_test)

# print('GB',confusion_matrix(y_test,gnb_predictions))
# print('GB',classification_report(y_test, gnb_predictions))
# print('GB',metrics.accuracy_score(y_test, gnb_predictions)*100)



# results = np.array(list(zip(data_test.id,predictions)))
# results = pd.DataFrame(results, columns=['id', 'target'])
# results.to_csv('LR_results.csv', index = False)

