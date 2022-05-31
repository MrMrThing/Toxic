import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_iris
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix


#Importing data
data = pd.read_excel('C:/Users/rasmu/Desktop/Train.xlsx', names = ['id', 'sentence', 'toxic', 'severe_toxic', 'obscene' , 'threat' , 'insult', 'identity_hate'])
test = pd.read_csv('C:/Users/rasmu/Desktop/Test.csv', names = ['id', 'comments'])
result = pd.read_csv('C:/Users/rasmu/Desktop/test_labels.csv', names = ['id','toxic', 'severe_toxic', 'obscene' , 'threat' , 'insult', 'identity_hate'])

#Making Dataframe's
df = pd.DataFrame(data)
df_result = pd.DataFrame(result)
df_test = pd.DataFrame(test)

#Selecting the sentences from the dataframe
sentences = df['sentence'].values
results = df_result['toxic'].values
test_sentences = df_test['comments'].values

#Here we define what label we want to work with
y = df['toxic'].values

#Info on the dataframe
df_result.head()
df_test.head()

#Splitting the data, test_size is how big in % the split is
sentences_train, sentences_test, y_train, y_test, = train_test_split(
    sentences,y,test_size=0.20,random_state=42)


#Test sentence for a quick prediction
test_sentence = ['the quick brown fox jumps over the lazy dog']

#Making an Count Vectorizer Template
#And fitting it to the sentences
vectorizer = CountVectorizer(lowercase=False)
vectorizer.fit(sentences_train)

#Transforming or training and test sets
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
X_train




#Transforming the test sentence
Test = vectorizer.transform(test_sentence)
X_test_sentences = vectorizer.transform(test_sentences)

#Checking the transforming was right
print(Test)

#setting an timer to see how long the training takes
start = time.time()

#Defining the used classifier, this can be changed
#And fitting that classifier on our training set
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Making the prediction on the test sentence
#And using the test split to give us a accuracy score
prediction = classifier.predict(Test)
score = classifier.score(X_test, y_test)

#Printing infomation to the user, and stopping the timer
print("Give me a sentence")

print("Accuracy:", score)
print("Prediction", prediction)

stop = time.time()
print(f"Training time: {stop - start}s")

#The rest is code to make the matrix

#Length of all the sentences
length = int(len(test_sentences))

#Going though the results and making them and int
#Some of them are strings
for i in range(1, length):
    results[i] = int(results[i])
    

#Variable holders for later
TP = 0
TN = 0
FP = 0
FN = 0

#Information to the user
print(length)

#Using k as an increment for showing progress
k = 0.05

#The loop for every sentence
#We start at one becouse the comments starts at the second row
for i in range(1, length-1):

    
    #This is just to show the progress of the whole process
    if i >= k * length:
        print(k*100, "%")
        k = k + 0.05

    #Making a prediction on a sentence and setting it to be and int
    #We start with +1 becouse the test set, starts one row behind the label set
    prediction = int(classifier.predict(X_test_sentences[i+1]))

    #Going though the scenarios and counting them 
    if prediction == 1 and results[i] == -1 or prediction == 1 and results[i] == 1:
        TP = TP + 1

    elif prediction == 0 and results[i] == 0:
        TN = TN + 1

    elif prediction == 1 and results[i] == 0:
        FN = FN + 1

    elif prediction == 0 and results[i] == -1 or prediction == 0 and results[i] == 1:
        FP = FP + 1

    #If none of the above went through something went wrong
    #So here is information on what row, and what the variables where
    else:
        print(i)
        print(prediction, results[i])
        break

#Ending by giving the information to the user
print("TP", TP/length)
print("TN", TN/length)
print("FP", FP/length)
print("FN", FN/length)

