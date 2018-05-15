import pandas as pd
import numpy as np

COLUMNS_NAME = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

DATA_PATH = ('../uci_data/adult.data.txt',
        '../uci_data/adult.test.txt')

def preprocess(data):
    # drop workclass feature
    del data['workclass']

    # convert sex feature into numerical 1 - Male, 0 - Female
    sex = data['sex']
    male = sex.str.contains('Male')
    data['sex'] = male.astype(int)

def load_data():
    train_data = pd.read_csv(DATA_PATH[0], header=None, names=COLUMNS_NAME)
    test_data = pd.read_csv(DATA_PATH[1], header=None, names=COLUMNS_NAME)

    return (train_data, test_data)

if __name__ == '__main__':
    train_data, test_data = load_data()
    preprocess(train_data)
    print(train_data.describe())