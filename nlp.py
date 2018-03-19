#nlp
import pandas as pd

#importing the dataset Delimiter--  ''' this is used as in tsv file columns are seperated by tab space '''  quoting-- '''to ignore doble quotes to create any problem'''
dataset = pd.read_csv('C:\\Users\\saksham raj seth\\Desktop\\sarc.tsv',delimiter = '\t',quoting=3)


SlangsDict = {
	r"\bv\b" : "we",r"\br\b" : "are",r"\bu\b" : "you",r"\bc\b" : "see",r"\by\b" : "why",r"\bb\b" : "be",r"\bda\b" : "the",r"\bhaha\b" : "ha",
	r"\bhahaha\b" : "ha",r"\bdon't\b" : "do not",r"\bdoesn't\b" : "does not",r"\bdidn't\b" : "did not",r"\bhasn't\b" : "has not",
	r"\bhaven't\b" : "have not",r"\bhadn't\b" : "had not",r"\bwon't\b" : "will not",r"\bwouldn't\b" : "would not",r"\bcan't\b" : "can not",
	r"\bcannot\b" : "can not", r"\bi'll\b" : "i will",r"\bwe'll\b" : "we will",r"\byou'll\b" : "you will",r"\bisn't\b" : "is not",
	r"\bthat's\b" : "that is",r"\bidk\b" : "i do not know",r"\btbh\b" : "to be honest",r"\bic\b" : "i see",r"\bbtw\b" : "by the way",
	r"\blol\b" : "laughing",r"\bimo\b" : "in my opinion"
}

"""
sentiment emotions dictionary
:D -> good
"""
sentiEmo = {
	"&lt;3" : " positive ",":D" : " positive ",	":d" : " positive ", ":dd" : " positive ", ":P" : " positive ", ":p" : " positive ","8)" : " positive ",
	"8-)" : " positive ",  ":-)" : " positive ",    ":)" : " positive ",    ";)" : " positive ",    "(-:" : " positive ",    "(:" : " positive ",
    ":')" : " positive ",    "xD" : " positive ",    "XD" : " positive ",  "yay!" : " positive ",  "yay" : " positive ",  "yaay" : " positive ",
    "yaaay" : " positive ",  "yaaaay" : " positive ", "yaaaaay" : " positive ", "Yay!" : " positive ", "Yay" : " positive ", "Yaay" : " positive ",
    "Yaaay" : " positive ", "Yaaaay" : " positive ", "Yaaaaay" : " positive ",  ":/" : " negative ", "&gt;" : " negative ", ":'(" : " negative ",
    ":-(" : " negative ", ":(" : " negative ", ":s" : " negative ",":-s" : " negative ","-_-" : " negative ", "-.-" : " negative "    
}




def replaceEmotions(Text):
    for key, value in sentiEmo.items():
        Text = Text.replace(key, value)
    return Text

def replaceSlangs(text):
    for r, slangs in SlangsDict.items():
        text = re.sub(r,slangs,text.lower()) #upper/lower case
    return text






#cleaning the texts
import re
#nltk.download('stopwords') #to download stopwords dictionary to remove neutral words
from nltk.corpus import stopwords #corpus is a word used to define same type of text
#stemming-- we want same words  like love- loved loving so it will be replaced by love basically words are replaced by their roots
from nltk.stem.porter import PorterStemmer
#from nltk.stem.lancaster import LancasterStemmer
import nltk
sno = nltk.stem.SnowballStemmer('english')
corpus = []
for i in range(0,1993):
	#review = 'Wow... Loved this place'
	#1st parameter what we want to remove----  ""^ --> means (not)""
	#2nd removed character replaced with
	#3rd parameter from where
    #review = re.sub('[^a-zA-Z]',' ',dataset['texts'][i])
    review = dataset['texts'][i]
    hashtags = re.compile(r'#\w*\s?')
    tags = re.compile(r'@\w*\s?')
    review = review.lower() #making everything in lower case
    output=''
    for text in review:
        text = re.sub(hashtags,'',text)
        text = re.sub(tags,'', text)
        output+=text
    review = output
    review = replaceEmotions(review)
    review = replaceSlangs(review)
    review = review.split()
    ps = PorterStemmer()
    #lcs=LancasterStemmer()
    #stoplist = 'for a an the or to his her he she them they of it'.split()      #word for word in review if word not in stoplist
    review = [sno.stem(word) for word in review if not word in set(stopwords.words('english'))] # ps.stem(word) for word in review if not word in set(stopwords.words('english')) # it will read all words in review list and if the word is not in stopwords list it will keep these words
    review = ' '.join(review) #for concatination of list to make it as a string  and space in starting is used for join every word with a space
    corpus.append(review)


#create the bag of words model -- created a sparse matrix of all unique words in complete corpus and will show their use in rows.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer() #cv = CountVectorizer(max_features = 1500) it will only take most frequent 1500 words
X = cv.fit_transform(corpus).toarray()#independent
y=dataset.iloc[: , 1].values#dependent

# copy from train_test_split  to   confusion matrix of any classification  but removed the standard scaler for feature scaling
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.21, random_state = 42)

# Fitting NaiveBAyes classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)

import numpy as np
sum1=cm1.sum()
diasum1=np.trace(cm1)

print('Naive Bayes accuracy:',end="")
print(diasum1*100/sum1)

#Fiiting K-NN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
classifier.fit(X_train, y_train)

y_pred2 = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred2)

sum2=cm2.sum()
diasum2=np.trace(cm2)

print('K-NN accuracy:',end="")
print(diasum2*100/sum2)


#SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

y_pred3 = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred3)

sum3=cm3.sum()
diasum3=np.trace(cm3)


print('SVM accuracy:',end="")
print(diasum3*100/sum3)

#Random Forest classification using Entropy
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

y_pred4 = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test, y_pred4)

sum4=cm4.sum()
diasum4=np.trace(cm4)

print('Random Forest accuracy:',end="")
print(diasum4*100/sum4)
'''
#Random Forest Classification using Gini Index
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 42)
classifier.fit(X_train, y_train)

y_pred5 = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm5 = confusion_matrix(y_test, y_pred5)


sum5=cm5.sum()
diasum5=np.trace(cm5)

print('Random Forest using Gini accuracy:',end="")
print(diasum5*100/sum5)'''