import random

class DataPrepocessing():
        
    @staticmethod
    def shuffle(dataset):
        dataset = dataset.sample(frac=1)
        return dataset

    @staticmethod
    def normalize(dataset):
        values = dataset.select_dtypes(exclude="object")
        columnNames = values.columns.tolist()
        columnNames.remove('label')
        for column in columnNames:
            dataset[column]/=255
        return dataset
    
    @staticmethod
    def split(dataset, training_len=0.70):
        training_len = int(len(dataset)*training_len)
        training_set = dataset.iloc[:training_len, :]
        testing_set = dataset.iloc[training_len:, :]
        return training_set, testing_set