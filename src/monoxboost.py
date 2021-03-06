"""
author: Jesús Enrique Cartas Rascón
repo: https://github.com/jesi-rgb/monotonic-class
"""

import codecs

import numpy as np
import pandas as pd

import xgboost as xgb

from scipy.io import arff

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

def split_and_train(data, target=None):
    '''Define label subsets for a given dataset. It figures out
    how many factors there are in the target value and
    subsets the labels based on that, yielding a new
    set of binary datasets. Those are then passed to
    a model to train, generating and returning the trained
    models. 
    
    > `data`: the dataset, optionally whithout the labels, and in ndarray form.
    
    > `target`: list for the labels of the data. If none, this function will take
    the last column of data as the target.
    
    > `model`: model to be used.
    '''

    # figure out how many factors there are in the target value
    unique_target = None
    if(target is None):
        unique_target = np.unique(data[:,-1])
    else:
        unique_target = np.unique(target)

    # create a list of binary subsets with monotonic constraints
    subsets = []
    for i in range(0, len(unique_target)):
        # binarize each dataset and save it to a list
        binary_target = np.where(target<=unique_target[i], 0, 1)
        binary_subset = np.array(list(zip(data, binary_target)), dtype='object')
        subsets.append(binary_subset)



    # define parameters for xgboost
    # evallist = [(dtest, 'eval'), (dtrain, 'train')]

    param = {'max_depth': 5, 'eta': 1}
    param['nthread'] = 8
    param['tree_method'] = 'exact'
    param['monotone_constraints'] = "1"

    num_round = 10
    
    
    # for each subset we train a model and save it
    models = []
    for subset in subsets:
        X = np.array(list(zip(*subset))[0])
        Y = np.array(list(zip(*subset))[1])
        dmatrix = xgb.DMatrix(X, label=Y)
        bst = xgb.train(param, dmatrix, num_boost_round = 1000)
        models.append(bst)

    # return the list of models trained in each binary subset
    return models


def predict_ensemble(data_points, labels, models):
    '''Computes the cascade prediction for all models, given
    a set of unknown datapoints.
    
    > `data_points`: a data matrix in the form of ndarray.

    > `labels`: a list of all the possible labels to be predicted.
    
    > `models`: a list of all the binary models generated by split and train
    '''

    # overwrite data_points to be a DMatrix
    data_points = xgb.DMatrix(data_points)

    # Instantiate a new dictionary which we'll save the final
    # results in.
    rows = dict()


    # for each model, we create a new entry with the predictions
    for i in range(1, len(labels)):
        rows.update({labels[i] : models[i].predict(data_points)})


    # create a dataframe out of the data and
    # calculate the sum to obtain the prediction
    df = pd.DataFrame.from_dict(rows)
    return 1 + df.sum(axis=1).round().to_numpy()

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


if __name__ == "__main__":
    # iris = load_iris().data
    # target = load_iris().target

    # example: load arff data
    data, target = load_arff("data/era.arff", target_index=-1)
    
    models = split_and_train(data, target=target)
    predictions = predict_ensemble(data, np.unique(target), models)
    print("\nScore: right/all answers is {}".format(np.count_nonzero(target == predictions)/len(target)))
