# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 23:29:34 2025

@author: jesus
"""

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import os
import pandas as pd
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
import re
from html.parser import HTMLParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers

X_train = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/archive/train.csv')
X_test = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/archive/test.csv')

print("Tamaño del conjunto de datos de entrenamiento: ", len(X_train))
print("Tamaño del conjunto de datos de pruebas: ", len(X_test))


print(X_train['target'].value_counts())

X_train['target'].hist()
plt.ylabel("# tweets")
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
# Calculamos el número de palabras
tweet_len_0 = X_train[X_train['target'] == 0]['text'].str.split().map(lambda x: len(x))
tweet_len_1 = X_train[X_train['target'] == 1]['text'].str.split().map(lambda x: len(x))
ax1.hist(tweet_len_0, color='green')
ax1.set_title('Non disaster tweets')
ax2.hist(tweet_len_1, color='red')
ax2.set_title('Disaster tweets')
fig.suptitle('Número de palabras por tweet')
plt.show()



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
# Calculamos el número de palabras
tweet_len_0 = X_train[X_train['target'] == 0]['text'].str.split().map(lambda x: len(set(x)))
tweet_len_1 = X_train[X_train['target'] == 1]['text'].str.split().map(lambda x: len(set(x)))
ax1.hist(tweet_len_0, color='green')
ax1.set_title('Non disaster tweets')
ax2.hist(tweet_len_1, color='red')
ax2.set_title('Disaster tweets')
fig.suptitle('Número de palabras por tweet')
plt.show()



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
# Calculamos el número de palabras
tweet_len_0 = X_train[X_train['target'] == 0]['text'].str.split().map(lambda x: np.mean([len(i) for i in x]))
tweet_len_1 = X_train[X_train['target'] == 1]['text'].str.split().map(lambda x: np.mean([len(i) for i in x]))
ax1.hist(tweet_len_0, color='green')
ax1.set_title('Non disaster tweets')
ax2.hist(tweet_len_1, color='red')
ax2.set_title('Disaster tweets')
fig.suptitle('Longitud media de las palabras por Tweet')
plt.show()



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
# Calculamos el número de caracteres por tweet
tweet_len_0 = X_train[X_train['target'] == 0]['text'].str.len()
tweet_len_1 = X_train[X_train['target'] == 1]['text'].str.len()
ax1.hist(tweet_len_0, color='green')
ax1.set_title('Non disaster tweets')
ax2.hist(tweet_len_1, color='red')
ax2.set_title('Disaster tweets')
fig.suptitle('Número caracteres por tweet')
plt.show()


stopwords.words('english')


def plot_stopwords(label):
    tweets_stopwords = {}
    for words in X_train[X_train['target'] == label]['text'].str.split():
        sw = list(set(words).intersection(stopwords.words('english')))
        for w in sw:
            if w in tweets_stopwords.keys():
                tweets_stopwords[w] += 1
            else:
                tweets_stopwords[w] = 1
                
    top = sorted(tweets_stopwords.items(), key=lambda x:x[1],reverse=True)[:10]
    plt.bar(*zip(*top))
    plt.show()


plot_stopwords(0)

plot_stopwords(1)



def plot_punctuation(label):
    tweets_stopwords = {}
    for words in X_train[X_train['target'] == label]['text'].str.split():
        sw = list(set(words).intersection(string.punctuation))
        for w in sw:
            if w in tweets_stopwords.keys():
                tweets_stopwords[w] += 1
            else:
                tweets_stopwords[w] = 1
                
    top = sorted(tweets_stopwords.items(), key=lambda x:x[1],reverse=True)[:20]
    plt.figure(figsize=(10, 5))
    plt.bar(*zip(*top))
    plt.show()




plot_punctuation(0)

plot_punctuation(1)




cv = CountVectorizer(ngram_range=(2, 2))
sum_words = cv.fit_transform(X_train['text']).sum(axis=0)

# Calculamos 
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)[:20]



plt.figure(figsize=(15, 7))
plt.barh(*zip(*words_freq))
plt.show()


def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)



remove_url("Esto es una prueba: http://localhost:8888/notebooks/Desktop/Workspace/Deep%20Neural%20Networks%20Course/11.%20Consideraciones%20de%20un%20proyecto%20de%20Deep%20Learning/code/Disaster%20Tweets.ipynb")





class HTMLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []
        
    def handle_data(self, d):
        self.fed.append(d)
        
    def get_data(self):
        return ''.join(self.fed)

def remove_html(text):
    s = HTMLStripper()
    s.feed(text)
    return s.get_data()



remove_html('<tr><td align="left"><a href="../../issues/51/16.html#article">Phrack World News</a></td>')

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)




remove_emoji("Omg another Earthquake 😔😔")



def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


remove_punctuation("hola #que tal")

X_train_prep = X_train.copy()

X_train_prep['text'] = X_train['text'].apply(remove_url)
X_train_prep['text'] = X_train['text'].apply(remove_html)
X_train_prep['text'] = X_train['text'].apply(remove_emoji)
X_train_prep['text'] = X_train['text'].apply(remove_punctuation)


X_test_prep = X_test.copy()

X_test_prep['text'] = X_test['text'].apply(remove_url)
X_test_prep['text'] = X_test['text'].apply(remove_html)
X_test_prep['text'] = X_test['text'].apply(remove_emoji)
X_test_prep['text'] = X_test['text'].apply(remove_punctuation)




cv = CountVectorizer(ngram_range=(2, 2))
sum_words = cv.fit_transform(X_train_prep['text']).sum(axis=0)

# Calculamos 
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)[:20]


plt.figure(figsize=(15, 7))
plt.barh(*zip(*words_freq))
plt.show()


Y_train = X_train_prep['target']




vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_prep['text'])


X_train = X_train.toarray()



X_test = vectorizer.transform(X_test_prep['text'])
X_test = X_test.toarray()





X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15)




print("Longitud subcojunto de entrenamiento: ", len(X_train))
print("Longitud subconjunto de validación: ", len(X_val))
print("Longitud subconjutno de pruebas: ", len(X_test))


model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(
    optimizer='adam', 
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision']
)



history = model.fit(
    X_train,
    Y_train,
    epochs=20,
    batch_size=1024,
    validation_data=(X_val, Y_val))





pd.DataFrame(history.history)[['loss', 'val_loss']].plot(figsize=(10, 7))
plt.grid(True)
plt.gca().set_ylim(0, 1.2)
plt.xlabel("epochs")
plt.show()



Y_pred = model.predict(X_test).round(0)
print(Y_pred)


for i in range(30):
    print("{} - {}".format(X_test_prep['text'][i], Y_pred[i]))




































