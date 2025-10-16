# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 09:09:29 2025

@author: jesus
"""

from html.parser import HTMLParser
import email
import string
import nltk
import os


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score


from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

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
    
    
def strip_tags(html):
    s = HTMLStripper()
    s.feed(html)
    return s.get_data()


t = '<tr><td align="left"><a href="../../issues/51/16.html#article">Phrack World News</a></td>'
strip_tags(t)



class Parser:

    def __init__(self):
        self.stemmer = nltk.PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.punctuation = list(string.punctuation)

    def parse(self, email_path):
        """Parse an email."""
        with open(email_path, errors='ignore') as e:
            msg = email.message_from_file(e)
        return None if not msg else self.get_email_content(msg)

    def get_email_content(self, msg):
        """Extract the email content."""
        subject = self.tokenize(msg['Subject']) if msg['Subject'] else []
        body = self.get_email_body(msg.get_payload(),
                                   msg.get_content_type())
        content_type = msg.get_content_type()
        # Returning the content of the email
        return {"subject": subject,
                "body": body,
                "content_type": content_type}

    def get_email_body(self, payload, content_type):
        """Extract the body of the email."""
        body = []
        if type(payload) is str and content_type == 'text/plain':
            return self.tokenize(payload)
        elif type(payload) is str and content_type == 'text/html':
            return self.tokenize(strip_tags(payload))
        elif type(payload) is list:
            for p in payload:
                body += self.get_email_body(p.get_payload(),
                                            p.get_content_type())
        return body

    def tokenize(self, text):
        """Transform a text string in tokens. Perform two main actions,
        clean the punctuation symbols and do stemming of the text."""
        for c in self.punctuation:
            text = text.replace(c, "")
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        tokens = list(filter(None, text.split(" ")))
        # Stemming of the tokens
        return [self.stemmer.stem(w) for w in tokens if w not in self.stopwords]
    
    
inmail = open("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/trec07p/data/inmail.1").read()
print(inmail)
    
    
p = Parser()
p.parse("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/trec07p/data/inmail.1")


index = open("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/trec07p/full/index").readlines()



DATASET_PATH = os.path.join("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets", "trec07p")

def parse_index(path_to_index, n_elements):
    ret_indexes = []
    index = open(path_to_index).readlines()
    for i in range(n_elements):
        mail = index[i].split(" ../")
        label = mail[0]
        path = mail[1][:-1]
        path_mail = path.split("/")[-1]
        ret_indexes.append({"label":label, "email_path":os.path.join(DATASET_PATH, os.path.join("data", path_mail))})
    return ret_indexes
    
    
    
    
def parse_email(index):
    p = Parser()
    pmail = p.parse(index["email_path"])
    return pmail, index["label"]
    
indexes = parse_index("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/trec07p/full/index", 10)
    
    
index = parse_index("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/trec07p/full/index", 1)
    
    
open(index[0]["email_path"]).read()


mail, label = parse_email(index[0])
print("El correo es:", label)
print(mail)




# Preapración del email en una cadena de texto
prep_email = [" ".join(mail['subject']) + " ".join(mail['body'])]

vectorizer = CountVectorizer()
X = vectorizer.fit(prep_email)

print("Features:\n", vectorizer.get_feature_names_out())




X = vectorizer.transform(prep_email)
print("\nValues:\n", X.toarray())


prep_email = [[w] for w in mail['subject'] + mail['body']]

enc = OneHotEncoder(handle_unknown='ignore')
X = enc.fit_transform(prep_email)

print("Features:\n", enc.get_feature_names_out())
print("\nValues:\n", X.toarray())


def create_prep_dataset(index_path, n_elements):
    X = []
    y = []
    indexes = parse_index(index_path, n_elements)
    for i in range(n_elements):
        print("\rParsing email: {0}".format(i+1), end='')
        try:
            mail, label = parse_email(indexes[i])
            X.append(" ".join(mail['subject']) + " ".join(mail['body']))
            y.append(label)
        except:
            pass
    return X, y


X_train, y_train = create_prep_dataset("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/trec07p/full/index", 100)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)



print(X_train.toarray())
print("\nFeatures:", len(vectorizer.get_feature_names_out()))


pd.DataFrame(X_train.toarray(), columns=[vectorizer.get_feature_names_out()])



nb_clf = BernoulliNB(alpha=1.0e-10)
nb_clf.fit(X_train, y_train)


X, y = create_prep_dataset("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/trec07p/full/index", 500)
X_test = X[100:]
y_test = y[100:]


X_test = vectorizer.transform(X_test)


y_pred = nb_clf.predict(X_test)


print("Predicción:\n", y_pred)
print("\nEtiquetas reales:\n", y_test)



print('F1 score: {:.3f}'.format(f1_score(y_test, y_pred, pos_label='ham')))


X, y = create_prep_dataset('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/trec07p/full/index', 22000)

    
X_train, y_train = X[:20000], y[:20000]
X_test, y_test = X[20000:], y[20000:]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
    
    
nb_clf = BernoulliNB(alpha=1.0e-10)
nb_clf.fit(X_train, y_train)


X_test = vectorizer.transform(X_test)


y_pred = nb_clf.predict(X_test)


print('F1 score: {:.3f}'.format(f1_score(y_test, y_pred, pos_label='ham')))



def plot_decision_boundary(X, y, clf):
    mins = X.min(axis=0) - 0.5
    maxs = X.max(axis=0) + 0.5  

    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], 71),
                     np.linspace(mins[1], maxs[1], 81))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    
    gs = gridspec.GridSpec(1, 2)
    gs.update(hspace=0.8)
    
    ax = plt.subplot(gs[0])
    ax.contourf(xx, yy, Z, cmap="RdBu", alpha=0.5)
    ax.plot(X[:, 0][y==1], X[:, 1][y==1], 'r.')
    ax.plot(X[:, 0][y==0], X[:, 1][y==0], 'b.')
    
    ax = plt.subplot(gs[1])
    ax.contour(xx, yy, Z, [0.5], colors='k')
    ax.plot(X[:, 0][y==1], X[:, 1][y==1], 'r.')
    ax.plot(X[:, 0][y==0], X[:, 1][y==0], 'b.')






iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.int64)



plt.figure(figsize=(12, 6))
plt.scatter(X[:,0][y == 0], X[:,1][y == 0], c="b", marker=".")
plt.scatter(X[:,0][y == 1], X[:,1][y == 1], c="r", marker=".")
plt.show()


clf = GaussianNB()
clf.fit(X, y)


plt.figure(figsize=(16, 5))
plot_decision_boundary(X, y, clf)
plt.show()





clf = LogisticRegression()
clf.fit(X, y)

plt.figure(figsize=(16,5))
plot_decision_boundary(X, y, clf)
plt.show()














    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    