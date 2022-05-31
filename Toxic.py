from asyncio.windows_events import NULL
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_iris

#Importing data
data = pd.read_excel('C:/Users/rasmu/Desktop/Train.xlsx', names = ['id', 'sentence', 'toxic', 'severe_toxic', 'obscene' , 'threat' , 'insult', 'identity_hate'])
test = pd.read_csv('C:/Users/rasmu/Desktop/Test.csv')

#Making Dataframe's
df = pd.DataFrame(data)
df_test = pd.DataFrame(test)
  
#Info on the dataframe
df.info()

#Selecting the sentences from the dataframe
sentences = df['sentence'].values

#Little function for erasing special characters from sentences  
"""
for i in range(len(sentences)):

    sentences[i] = re.sub('\W+',' ', sentences[i] )

    if i >= k * len(sentences):
        print(k*100, "%")
        k = k + 0.1

    print(sentences[i])

print("100%")
"""

#Here we define what label we want to work with
y = df['toxic'].values

#Splitting the data, test_size is how big in % the split is
sentences_train, sentences_test, y_train, y_test = train_test_split(
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

#Checking the transforming was right
print(Test)

#setting an timer to see how long the training takes
start = time.time()

#Defining the used classifier, this can be changed
#And fitting that classifier on our training set
classifier = DecisionTreeClassifier()
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

#function to predict a full test set
def predictDataSet(DataFrame):

    #Using k as an increment for showing progress
    k = 0.1

    #making an dataframe that will be used later
    df_predictions = pd.DataFrame(columns = ['id', 'statement', 'prediction'])
    
    #Length of the dataframe, and making an array to hold the predictions
    length = int(len(DataFrame))
    print(length)
    allPredictions = []

    #The loop for every datapoint in the dataframe
    for i in range(length):

        #Taking a sentence from the dataframe
        test_sentence = DataFrame.iloc[i]

        #Transforming and predicting the sentence
        Test = vectorizer.transform(test_sentence)
        prediction = classifier.predict(Test)

        #Inserting the prediction into the array
        allPredictions.append(prediction)
        test_sentence = NULL

        #This is just to show the progress of the whole process
        if i >= k * length:
            print(k*100, "%")
            k = k + 0.1

    #When the loop is done, we take all the predictions and setting them into the dataframe
    #Then exporting that dataframe out
    DataFrame.insert(len(DataFrame.columns), 'predictions', allPredictions)
    print(DataFrame)
    DataFrame.to_csv('Predicted_File.csv', sheet_name='new_sheet_name')
    return NULL

#Calling the functing, needs a dataframe to predict
predictDataSet(df_test)