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

data = pd.read_excel('C:/Users/rasmu/Desktop/Train.xlsx', names = ['id', 'sentence', 'toxic', 'severe_toxic', 'obscene' , 'threat' , 'insult', 'identity_hate'])

test = pd.read_excel('C:/Users/rasmu/Desktop/Test.xlsx')

df = pd.DataFrame(data)

df_test = pd.DataFrame(test)
  
df.info()


sentences = df['sentence'].values

"""

for i in range(len(sentences)):

    sentences[i] = re.sub('\W+',' ', sentences[i] )

    if i >= k * len(sentences):
        print(k*100, "%")
        k = k + 0.1

    print(sentences[i])

print("100%")
"""

y = df['toxic'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences,y,test_size=0.20,random_state=42)



test_sentence = ['the quick brown fox jumps over the lazy dog']

vectorizer = CountVectorizer(lowercase=False)
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
X_train

Test = vectorizer.transform(test_sentence)

print(Test)

start = time.time()

classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

prediction = classifier.predict(Test)
score = classifier.score(X_test, y_test)

print("Give me a sentence")

print("Accuracy:", score)
print("Prediction", prediction)

stop = time.time()
print(f"Training time: {stop - start}s")

"""
iris = load_iris()
sentences, y = iris.data, iris.target
tree.plot_tree(classifier)
"""
def predictDataSet(DataFrame):

    k = 0.1
    df_predictions = pd.DataFrame(columns = ['id', 'statement', 'prediction'])
    
    length = int(len(DataFrame))
    print(length)
    allPredictions = []

    for i in range(length):

        test_sentence = DataFrame.iloc[i]

        Test = vectorizer.transform(test_sentence)
        prediction = classifier.predict(Test)
        print(i)

        allPredictions.append(prediction)

        #df_predictions = df.append(DataFrame.iloc[i], DataFrame.iloc[i], prediction)

        test_sentence = NULL
        if i >= k * length:
            print(k*100, "%")
            k = k + 0.1

    DataFrame.insert(len(DataFrame.columns), 'predictions', allPredictions)
    print(DataFrame)
    DataFrame.to_excel('pandas_to_excel.xlsx', sheet_name='new_sheet_name')
    return NULL

predictDataSet(df_test)