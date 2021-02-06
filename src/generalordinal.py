"""
author: Jesús Enrique Cartas Rascón
repo: https://github.com/jesi-rgb/monotonic-class
"""

import codecs
from operator import mul, sub

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

'''
Se implementará una función que recibirá un data set 
como parámetro y será capaz de formar los data sets 
binarios derivados a partir del número de clases del 
data set original. 

Así mismo, lanzará un clasificador diferente al J48 
(a elección por el estudiante) para cada uno y 
devolverá una colección de modelos construidos.

Una segunda función recibirá dicha colección y 
será capaz de predecir un ejemplo (o un conjunto 
de ejemplos) a partir de los mismos usando la 
cascada de probabilidades que define el modelo 
múltiple.
'''


def split_and_train(data, target=None, model=None):
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

    # if no model is provided, we propose a Tree Classifier
    if(model is None):
        model = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)

    # for each subset we train a model and save it
    models = []
    for subset in subsets:
        X = list(zip(*subset))[0]
        Y = list(zip(*subset))[1]
        fit_model = model.fit(X, Y)
        models.append(fit_model)

    # return the list of models trained in each binary subset
    return models

def predict_ensemble(data_points, labels, models, operator=mul):
    '''Computes the cascade prediction for all models, given
    a set of unknown datapoints.
    
    > data_points: a data matrix in the form of ndarray.

    > labels: a list of all the possible labels to be predicted.
    
    > models: a list of all the binary models generated by split and train

    > operator: mul or sub. Not passed as string.
    Choose between the operator used since the authors of 
    the technique released two different papers suggesting the 
    difference and product operator, with varying results.
    '''

    # Instantiate a new dictionary which we'll save the final
    # results in.
    rows = dict()

    # Special case if there's only 3 labels (and two models) to compute
    if(len(models) == 2):
        # Taking v[:,0] is the same as saying 1-v[:,1] as stated
        # in the original paper, so we just take the first index.
        p_class_1 = models[0].predict_proba(data_points)[:,0]
        p_class_2 = 1 - models[1].predict_proba(data_points)[:,1]
        p_class_3 = models[1].predict_proba(data_points)[:,1]
        
        # Update the dictionary with the data to be processed
        rows = {labels[0]: p_class_1, labels[1]: p_class_2, labels[2]: p_class_3}
        

    # Otherwise
    else:
        #first case, first label and model
        rows.update({labels[0] : models[0].predict_proba(data_points)[:,0]})

        # middle cases
        for i in range(1, len(labels)-1):
            rows.update({labels[i] : operator(models[i-1].predict_proba(data_points)[:,1], models[i].predict_proba(data_points)[:,0])})

        #last case
        rows.update({labels[-1] : models[-1].predict_proba(data_points)[:,1]})


    # Create a dataframe with all the probabilities, and 
    # then compute the maximum. idxmax very conveniently
    # returns the column name that was the maximum, instead
    # of the value itself. Then just return it as a ndarray.
    df = pd.DataFrame.from_dict(rows)
    return df.idxmax(axis=1).to_numpy()


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

    # load arff data
    data, target = load_arff("Material/esl.arff", target_index=-1)

    # split the data and train the models
    models = split_and_train(data, target=target)

    # make cascading prediction
    predictions = predict_ensemble(data, np.unique(target), models, operator=sub)

    # very basic score measure
    print("Score: right/all answers is {}".format(np.count_nonzero(target == predictions)/len(target)))
