import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

data1 = pd.read_csv('dane\spam_ham_dataset.csv')
data1.drop('Unnamed: 0', inplace=True, axis=1)
data1.drop('label', inplace=True, axis=1)
data1['text'] = data1['text'].astype("string")
#print(data1['text'].dtype)
#print(data1['label_num'].dtype)
print(data1)


data2 = pd.read_csv('dane\enron_spam_data.csv')
data2 = data2.rename(columns = {"Spam/Ham":"label_num"})
data2 = data2.rename(columns = {"Message":"text"})
data2['label_num'] = data2['label_num'].replace('ham',int(0))
data2['label_num'] = data2['label_num'].replace('spam',int(1))

data2['text'] = data2['text'].astype("string")
pd.to_numeric(data2["label_num"])

#print(data2['text'].dtype)
#print(data2['label_num'].dtype)
data2.drop('Subject', inplace=True, axis=1)
data2.drop('Date', inplace=True, axis=1)
data2.drop(data2.columns[data2.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
print(data2)

data1.drop_duplicates(inplace = True)
data2.drop_duplicates(inplace = True)


data_final = pd.concat([data1, data2], axis=0)
data_final = data_final.dropna()
print(data_final)


nltk.download('stopwords') 

stopwords = stopwords.words('english')

punctuation = list(string.punctuation)


def filter_text(text):
  no_punctuation = [ch for ch in text if ch not in punctuation]
  no_punctuation = ''.join(no_punctuation)
  filtered_words = [word for word in str(no_punctuation).split() if word.lower() not in stopwords]
  return filtered_words



data_final['text'].apply(filter_text)

count_words = CountVectorizer(lowercase=True, analyzer=filter_text).fit_transform(data_final['text'])

x_train, x_test, y_train, y_test = train_test_split(count_words, data_final['label_num'], test_size=0.25, random_state = 42)

nb = MultinomialNB()
classifier = nb.fit(x_train, y_train)

preds = nb.predict(x_train)

print(preds)
print(y_train.values)

print(classification_report(y_train, preds))

print("Tablica pomyłek:\n", confusion_matrix(y_train, preds))
print("\n Dokładność: ", accuracy_score(y_train, preds))

print(nb.predict(x_test))
print(y_test.values)

print("Tablica pomyłek:\n", confusion_matrix(y_test, nb.predict(x_test)))
print("\n Dokładność: ", accuracy_score(y_test,nb.predict(x_test)))





