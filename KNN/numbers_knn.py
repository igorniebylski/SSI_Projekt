import pandas as pd
from datapreprocessing import DataPrepocessing
import random
import math

class KNN:
    @staticmethod
    def minkowski_distance(x, y, m):
        n = len(x)
        sum = 0
        for i in range(n):
            sum+=math.pow(abs(x[i]-y[i]),m)
        sum=math.pow(sum,1/m)
        return sum
    
    @staticmethod
    def get_accuracy(learn, test, k, m):
        X_train = learn.drop('label', axis=1)
        X_test = test.drop('label', axis=1)

        y_train = learn.label
        y_test = test.label

        correct_answers = 0

        for i in range(len(X_test)):
            choices = dict.fromkeys(learn.label.unique(), 0)
            distances = list()
            tmp = X_test.iloc[i]
            for j in range(len(X_train)):
                distances.append(KNN.minkowski_distance(tmp, X_train.iloc[j], m))
            distances = pd.DataFrame(data=distances, columns=['distance'], index=y_train.index)
            distances = distances.sort_values(by=['distance'], axis=0)[:k]
            
            for x in distances.index:
                choices[y_train.loc[x]]+=1
                
            max_choices=[]
            for element in choices:
                m = max(choices.values())
                if choices[element]==m:
                    max_choices.append(element)
            
            result = random.choice(max_choices)
            ans = y_test.iloc[i]
            if result==ans:
                correct_answers+=1
        return (correct_answers/len(y_test))*100

if __name__ == "__main__":
    data = pd.read_csv('numbers_smaller.csv')
    data = data.reset_index(drop=True)
    data = DataPrepocessing.normalize(data)
    data = DataPrepocessing.shuffle(data)
    l = DataPrepocessing.split(data, 0.7)
    training_set = l[0]
    testing_set = l[1]
    
    
    print(KNN.get_accuracy(training_set, testing_set, 5, 3))