import pandas as pd
import numpy as np
from functools import partial
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from skopt import gp_minimize
from skopt import space
from xgboost import XGBClassifier
import config


def optimize(params, param_names, x, y):
    """Takes all arguments from search space and traning features/target
        Initializes the models by setting the chosen param and runs CV
    Args:
        params [dict]: convert params to dict
        param_names [list]: make a list of param names
        x [float]: feature values
        y [int]: target values as binary
    Returns:
        [float]: Returns an accuracy score after 5 Folds
    """
    # set the parameters as dictionaries
    params = dict(zip(param_names, params))

    # initiate XGBClassifier and K-fold (5)
    model = XGBClassifier(**params)
    kf = StratifiedKFold(n_splits=5)
    acc = []

    # loop over kfolds
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        pred = model.predict(xtest)

        # append mean-accuracy to empty list
        fold_accuracy = accuracy_score(ytest, pred)
        acc.append(fold_accuracy)
    # return negative acc to find max optimization
    return -np.mean(acc)


df = pd.read_csv(config.CLEAN_FILE)
targets = df["response"].values
features = df.drop("response", axis=1).values

# define the range of input values to test the Bayes_op to create prop-distribution
param_space = [
    space.Integer(4, 24, name="max_depth"),
    space.Integer(1, 9, name="gamma"),
    space.Integer(20, 150, name="reg_alpha"),
    space.Real(0.01, 1, prior="uniform", name="reg_lambda"),
    space.Integer(1, 10, name="min_child_weight"),
    space.Real(0.05, 0.30, prior="uniform", name="eta"),
    space.Real(0.5, 1, prior="uniform", name="colsample_bytree"),
    space.Real(0.6, 0.95, prior="uniform", name="base_score"),
]

param_names = [
    "max_depth",
    "gamma",
    "reg_alpha",
    "reg_lambda",
    "min_child_weight",
    "eta",
    "colsample_bytree",
    "base_score",
]

# define the loss function to minimize (acc will be negative)
optimization_function = partial(
    optimize, param_names=param_names, x=features, y=targets
)

# initiate gp_minimize for Bayesian Optimization to select the best input values
result = gp_minimize(
    optimization_function,
    dimensions=param_space,
    n_calls=10,
    n_random_starts=10,
    verbose=10,
)
print(dict(zip(param_names, result.x)))
