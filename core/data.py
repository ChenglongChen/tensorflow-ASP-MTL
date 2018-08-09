
import os
import numpy as np


class DataLoader(object):

    @staticmethod
    def _load_inner(file_name):
        X = []
        y = []
        with open(file_name, "r") as f:
            for line in f:
                try:
                    terms = line.strip().split("\t")
                    if len(terms) == 2:
                        label = int(terms[0])
                        words = terms[1].strip().split(" ")
                        y.append(label)
                        X.append(words)
                except:
                    pass
        return X, np.array(y, dtype=np.int32)


    @staticmethod
    def load(data_dir, task_names):
        X_train = {}
        y_train = {}
        X_test = {}
        y_test = {}
        for task_name in task_names:
            train_file = os.path.join(data_dir, task_name + '.task.train')
            X_train[task_name], y_train[task_name] = DataLoader._load_inner(train_file)
            test_file = os.path.join(data_dir, task_name + '.task.test')
            X_test[task_name], y_test[task_name] = DataLoader._load_inner(test_file)
        return X_train, y_train, X_test, y_test
