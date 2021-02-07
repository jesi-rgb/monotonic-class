"""
author: Jesús Enrique Cartas Rascón
repo: https://github.com/jesi-rgb/monotonic-class
"""

import codecs
from operator import mul, sub

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.io import arff
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def split_and_train(data, target=None):
    '''Define subsets for a given dataset. It figures out
    how many factors there are in the target value and
    subsets the original dataset based on that, returning
    the subsets.
    
    > data: the dataset, specifically whithout the labels, and in ndarray form.
    
    > target: list for the labels of the data. If none, this function will take
    the last column of data as the target.
    
    > model: model to be used. Not tested for other kinds of models.
    '''

    # figure out how many factors there are in the target value
    if(target is None):
        unique_target = np.unique(data[:,-1])
    unique_target = np.unique(target)

    # create a list of subsets, for each factor except for
    # the last one
    subsets = []
    for i in range(0, len(unique_target)-1):
        # binarize each dataset and save it to a list
        binary_target = np.where(target<=unique_target[i], 0, 1)
        binary_subset = np.array(list(zip(data, binary_target)), dtype='object')
        subsets.append(binary_subset)



    # define parameters for xgboost
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    param = {'max_depth': 2, 'eta': 1}
    param['nthread'] = 4
    param['tree_method'] = 'exact'
    param['monotone_constraints'] = "(0, 1)"
    param['eval_metric'] = 'auc'

    num_round = 10
    
    # for each subset we train a model and save it
    models = []
    for subset in subsets:
        X = list(zip(*subset))[0]
        Y = list(zip(*subset))[1]
        dtrain = xgb.DMatrix(X, label=Y)
        bst = xgb.train(param, dtrain, num_round, evallist)
        models.append(bst)

    # return the list of models trained in each binary subset
    return models

def load_arff(path, target_index=-1):
    '''
    Helper function to parse arff data. This function will
    take the .arff file from `path` and extract the target
    column via `target_index`. By default is set as the
    last column.

    > `path`: string. Path to the .arff file.

    > `target_index`: int. Index of the column in which the
    target variable lives.

    > returns the data and the target in different variables
    in ndarray form.
    '''

    # helper functions from scipy
    df = None
    with codecs.open(path, 'r', encoding='utf8') as raw_data:
        arff_data = arff.loadarff(raw_data)
        df = pd.DataFrame(arff_data[0])

    # take the target column    
    target = df.iloc[:,target_index].to_numpy()

    # take everything but the target column
    data = df.drop(df.columns[target_index], axis=1).to_numpy()

    return data, target

def train_xgboost(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
    
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'

    num_round = 10
    bst = xgb.train(param, dtrain, num_round, evallist)

    return bst


if __name__ == "__main__":
    # iris = load_iris().data
    # target = load_iris().target

    # load arff data
    data, target = load_arff("data/era.arff", target_index=-1)
    
    bst = train_xgboost(data, target)