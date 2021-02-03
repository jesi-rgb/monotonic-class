import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.datasets import load_iris

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
    the subsets.'''

    # figure out how many factors there are in the target value
    unique_target = np.unique(target)

    # create a list of subsets, for each factor except for
    # the last one
    subsets = []
    for i in range(0, len(unique_target)-1):
        # binarize each dataset and save it to a list
        binary_target = np.where(target<=unique_target[i], 0, 1)
        binary_subset = np.array(list(zip(data, binary_target)), dtype='object')
        subsets.append(binary_subset)

    print(subsets)

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
    return dict(zip(unique_target, models))

def predict_ensemble(data_points, named_models):
    k = len(named_models)
    # first model
    # first_model = models[0].predict_proba(data_points)
    # print(1-first_model[0][0])
    names = named_models.keys
    models = named_models.values

    if(k==2):
        pass
        # predict_class_1 = 1-models[0].predict_proba(data_points)[:,0]
        # predict_class_2 = models[1].predict_proba(data_points)[:,0]
        # a = list(zip(predict_class_1, predict_class_2))



    else:
        pass
        # predictions = [1-model.predict_proba(data_points)[:,0] for model in models]
        # print(predictions)






if __name__ == "__main__":
    data = load_iris().data
    target = load_iris().target

    models = split_and_train(data, target=target)
    predict_ensemble(data, models)