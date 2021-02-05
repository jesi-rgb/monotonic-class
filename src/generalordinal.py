import numpy as np
import pandas as pd
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

def predict_ensemble(data_points, labels, models):

    if(len(models) == 2):
        p_class_1 = models[0].predict_proba(data_points)[:,0]
        p_class_2 = 1 - models[1].predict_proba(data_points)[:,1]
        p_class_3 = models[1].predict_proba(data_points)[:,1]
        
        rows = {labels[0]: p_class_1, labels[1]: p_class_2, labels[2]: p_class_3}
        df = pd.DataFrame.from_dict(rows)
        return df.idxmax(axis=1)

    else:
        rows = dict()
        #first case
        rows = rows.update({labels[0]: models[0].predict_proba(data_points)[:,0]})

        # middle cases (change * for -)
        for i in range(1, len(labels)-2):
            rows.update({labels[i]: models[i-1].predict_proba(data_points)[:,1] * models[i].predict_proba(data_points)[:0]})

        #last case
        rows.update({labels[len(labels)]: models[len(labels)-1].predict_proba(data_points)[:,1]})

        print(rows)
        

        







if __name__ == "__main__":
    data = load_iris().data
    target = load_iris().target

    models = split_and_train(data, target=target)
    predict_ensemble(data, np.unique(target), models)