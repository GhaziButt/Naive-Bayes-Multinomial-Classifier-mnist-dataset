import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
import tensorflow as tf
import random
import operator
from sklearn import metrics

(X_T , Y_T) , (X_TE , Y_TE) =tf.keras.datasets.mnist.load_data()

classifier = MultinomialNB()

X_T = X_T.reshape((60000 , 784)) 
X_TE = X_TE.reshape((10000 ,784))

classifier.fit(X_T, Y_T)

y_pred = classifier.predict(X_TE)

print("Accuracy using Naive Baye's Multinomial  is :" , metrics.accuracy_score(Y_TE ,y_pred)*100)