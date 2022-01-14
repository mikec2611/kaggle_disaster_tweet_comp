import kaggle
import nltk
import pandas as pd
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


import warnings
warnings.filterwarnings('ignore')

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
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# data_train = data_train.head()
# data_test = data_test.head()

# view data
# print(data_train.describe())
# print(data_test.describe())
# print(data_train.head())
# print(data_test.head())

# clean text
# lowering the text
data_train.text = data_train.text.apply(lambda x:x.lower() )
data_test.text = data_test.text.apply(lambda x:x.lower())
#removing square brackets
data_train.text = data_train.text.apply(lambda x:re.sub('\[.*?\]', '', x) )
data_test.text = data_test.text.apply(lambda x:re.sub('\[.*?\]', '', x) )
data_train.text = data_train.text.apply(lambda x:re.sub('<.*?>+', '', x) )
data_test.text = data_test.text.apply(lambda x:re.sub('<.*?>+', '', x) )
#removing hyperlink
data_train.text = data_train.text.apply(lambda x:re.sub('https?://\S+|www\.\S+', '', x) )
data_test.text = data_test.text.apply(lambda x:re.sub('https?://\S+|www\.\S+', '', x) )
#removing puncuation
data_train.text = data_train.text.apply(lambda x:re.sub('[%s]' % re.escape(string.punctuation), '', x) )
data_test.text = data_test.text.apply(lambda x:re.sub('[%s]' % re.escape(string.punctuation), '', x) )
data_train.text = data_train.text.apply(lambda x:re.sub('\n' , '', x) )
data_test.text = data_test.text.apply(lambda x:re.sub('\n', '', x) )
#remove words containing numbers
data_train.text = data_train.text.apply(lambda x:re.sub('\w*\d\w*' , '', x) )
data_test.text = data_test.text.apply(lambda x:re.sub('\w*\d\w*', '', x) )



# tokenize corpora
data_train.text = data_train.text.apply(lambda x:word_tokenize(x))
data_test.text = data_test.text.apply(lambda x:word_tokenize(x))

# print('token | ', data_train.text[0])

# remove stopwords
stop_words = set(stopwords.words('english'))
data_train.text = data_train.text.apply(lambda x: [w for w in x if not w in stop_words])
data_train.test = data_test.text.apply(lambda x: [w for w in x if not w in stop_words])

# print('remove stopwords | ', data_train.text[0])

# stem
stemmer = SnowballStemmer('english')
data_train.text = data_train.text.apply(lambda x:" ".join(stemmer.stem(token) for token in x))
data_test.text = data_test.text.apply(lambda x:" ".join(stemmer.stem(token) for token in x))

# print('stemmed | ', data_train.text[0])


X_train = data_train.text
y_train = data_train.target
X_test = data_test.text

# vectorize text
count_vectorizer = CountVectorizer()
X_train = count_vectorizer.fit_transform(X_train)
X_test = count_vectorizer.transform(X_test)

# train model
CLR = LogisticRegression(C=2)
# scores = cross_val_score(CLR, train_vectors_count, data_train.target, cv=6, scoring="f1")
CLR.fit(X_train, y_train)

# predict on test data
pred = CLR.predict(X_test)


results = np.array(list(zip(data_test.id,pred)))
results = pd.DataFrame(results, columns=['id', 'target'])
results.to_csv('results.csv', index = False)

# # evaluate results
# print('MAE:', metrics.mean_absolute_error(y_train, pred))
# print('MSE:', metrics.mean_squared_error(y_train, pred))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, pred)))
