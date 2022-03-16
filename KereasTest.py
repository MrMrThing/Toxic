import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_excel('C:/Users/rasmu/Desktop/Train.xlsx', names = ['id', 'sentence', 'toxic', 'severe_toxic', 'obscene' , 'threat' , 'insult', 'identity_hate'])

df = pd.DataFrame(data)
  
print(df)


sentences = df['sentence'].values
y = df['insult'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences,y,test_size=0.20,random_state=1000)


vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
X_train


classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)



print("Accuracy:", score)