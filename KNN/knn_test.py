from numbers_knn import *
from datetime import datetime

def main():
    test_result=dict()
    for i in range(0,10):
        data = pd.read_csv('numbers_smaller.csv')
        data = data.reset_index(drop=True)
        data = DataPrepocessing.normalize(data)
        data = DataPrepocessing.shuffle(data)
        prepared_data = DataPrepocessing.split(data, 0.70)
        learn = prepared_data[0]
        test = prepared_data[1]
            
        for k in (3,5):
            for m in (2,3,4,5):
                d1 = datetime.now()
                res= KNN.get_accuracy(learn, test, k, m)
                d2 = datetime.now()
                if (k,m) not in test_result.keys():
                    test_result[(k,m)]=[]
                test_result[(k,m)].append([res,d2-d1])

    # srednia i odchylenie
    print('(k,m), średnia dokładność i odchylenie, średni czas i odchylenie')
    for index in test_result.keys():
        avg_acc = sum(float(acc) for acc,time in test_result[index])/len(test_result[index])
        avg_time = sum(float(time.total_seconds()) for acc,time in test_result[index])/len(test_result[index])
        dev_acc=0
        dev_time=0
        for acc,time in test_result[index]:
            dev_acc+=(acc-avg_acc)**2
            dev_time+=(float(time.total_seconds())-avg_time)**2
        dev_acc=math.sqrt(dev_acc/2)
        dev_time=math.sqrt(dev_time/2)
        print(index, avg_acc, dev_acc, avg_time, dev_time)

if __name__ == "__main__":
    main()