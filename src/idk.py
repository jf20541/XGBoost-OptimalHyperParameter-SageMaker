
# import the libraries needed
import xgboost as xgb
import numpy as np
import pandas as pd
import sklearn

from sklearn.datasets import load_iris
iris = load_iris()


data = pd.DataFrame(iris.data)
data.columns = iris.feature_names
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this datset
num_round = 20  # the number of training iterations


bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)

import numpy as np
best_preds = np.asarray([np.argmax(line) for line in preds])

from sklearn.metrics import precision_score, f1_score
print(precision_score(y_test, best_preds, average='macro'))
print(f1_score(y_test, best_preds, average='weighted'))