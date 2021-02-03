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
    print(unique_target)

    # create a list of subsets, for each factor except for
    # the last one
    subsets = []
    for i in range(0, len(unique_target)-1):
        # binarize each dataset and save it to a list
        binary_target = np.where(target==unique_target[i], 0, 1)
        binary_subset = np.array(list(zip(data, binary_target)), dtype='object')
        subsets.append(binary_subset)

    if(model is None):
        model = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)

    
    models = []
    for subset in subsets:
        X = list(zip(*subset))[0]
        Y = list(zip(*subset))[1]
        fit_model = model.fit(X, Y)
        models.append(fit_model)

    return models




# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
# clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
# clf_entropy.fit(X_train, y_train)
# probs = clf_entropy.predict_proba(X_test)


if __name__ == "__main__":
    data = load_iris().data
    target = load_iris().target

    split_and_train(data, target=target)
    predict(data, models)