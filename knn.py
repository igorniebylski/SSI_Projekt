import statistics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import random as r
import time
import math

dataset = pd.read_csv("C:\\Users\\igorn\\Desktop\\projekt\\numbers_smaller.csv").copy()
X=dataset.drop('label', axis=1)
def KNN():
    accuracy = []
    timer = []
    y=dataset.label
    for i in range(100):
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
    avg_acc = sum(accuracy)/len(accuracy)
    avg_time = sum(timer)/len(timer)
    dev_acc=0
    dev_time=0
    for acc in accuracy:
        dev_acc+=(acc-avg_acc)**2
    for t in timer:
        dev_time+=(float(t)-avg_time)**2
        
    dev_acc=math.sqrt(dev_acc/2)
    dev_time=math.sqrt(dev_time/2)
    print(avg_acc, dev_acc, avg_time, dev_time)

KNN()
