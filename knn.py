import statistics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import random as r
import time

dataset = pd.read_csv("C:\\Users\\igorn\\Desktop\\projekt\\numbers_smaller.csv").copy()
X=dataset.drop('label', axis=1)
def KNN():
    accuracy = []
    timer = []
    y=dataset.label
    for i in range(10):
        time_start = time.time()
        n = r.randint(1, 15900)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=n)
        K=knn()

        # Training the model on the training data and labels
        K.fit(x_train,y_train)

        # Using the model to predict the labels of the test data
        y_pred = K.predict(x_test)

        # Evaluating the accuracy of the model using the sklearn functions
        accuracy.append(accuracy_score(y_test,y_pred)*100)
        confusion_mat = confusion_matrix(y_test,y_pred)
        timer.append(time.time() - time_start)
    print("mean accuracy", str(sum(accuracy)/len(accuracy)) + "\nmean time: " + str(sum(timer) / len(timer)))


KNN()
