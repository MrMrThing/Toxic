import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer

start = time.time()

data = pd.read_excel('C:/Users/rasmu/Desktop/Train.xlsx', names = ['id', 'sentence', 'toxic', 'severe_toxic', 'obscene' , 'threat' , 'insult', 'identity_hate'])

df = pd.DataFrame(data)
  
print(df)


sentences = df['sentence'].values
y = df['toxic'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences,y,test_size=0.20,random_state=1000)


vectorizer = CountVectorizer(lowercase=False)
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
X_train

print("Hello")

classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)



print("Accuracy:", score)

stop = time.time()
print(f"Training time: {stop - start}s")