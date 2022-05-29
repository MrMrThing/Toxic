from asyncio.windows_events import NULL
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import re
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import numpy as np



data = pd.read_excel('C:/Users/rasmu/Desktop/Train.xlsx', names = ['id', 'sentence', 'toxic', 'severe_toxic', 'obscene' , 'threat' , 'insult', 'identity_hate'])
test = pd.read_excel('C:/Users/rasmu/Desktop/Test.xlsx', names = ['id', 'comments'])
result = pd.read_csv('C:/Users/rasmu/Desktop/test_labels.csv', names = ['id','toxic', 'severe_toxic', 'obscene' , 'threat' , 'insult', 'identity_hate'])

df = pd.DataFrame(data)
df_result = pd.DataFrame(result)
df_test = pd.DataFrame(test)

sentences = df['sentence'].values
results = df_result['toxic'].values
test_sentences = df_test['comments'].values

y = df['toxic'].values

df_result.head()

df_test.head()


sentences_train, sentences_test, y_train, y_test, = train_test_split(
    sentences,y,test_size=0.20,random_state=42)


test_sentence = ['the quick brown fox jumps over the lazy dog']

vectorizer = CountVectorizer(lowercase=False)
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
X_train

X_test_sentences = vectorizer.transform(test_sentences)

Test = vectorizer.transform(test_sentence)

print(Test)

start = time.time()

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

prediction = classifier.predict(Test)
score = classifier.score(X_test, y_test)

print("Give me a sentence")

print("Accuracy:", score)
print("Prediction", prediction)

stop = time.time()
print(f"Training time: {stop - start}s")

#np.isnan(df_test.any()) #and gets False
#np.isfinite(df_result.all()) #and gets True

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

#plot_confusion_matrix(classifier, X_test_sentences, results)
#plt.show()

TP = 0
TN = 0
FP = 0
FN = 0

length = int(len(test_sentences))

for i in range(length):
    prediction = classifier.predict(X_test_sentences[i])
    if prediction == -1 and results[i] == 1:
        TP = TP + 1
    if prediction == 0 and results[i] == 0:
        TN = TN + 1
    if prediction == -1 and results[i] == 0:
        FN = FN + 1
    if prediction == 0 and results[i] == 1:
        FP = FP + 1

print("TP", TP)
print("TN", TN)
print("FP", FP)
print("FN", FN)
