from sklearn.model_selection import StratifiedKFold
import pandas as pd
import config
from pprint import pprint
import xgboost as xgb

"""
Dataset doesn't have equal distribution of targets values
use Stratified Cross-Validation for Imbalanced Classification, 
maintains the same class distribution in each subset
"""

df = pd.read_csv(config.CLEAN_FILE)
targets = df["response"]
print(
    f"Imbalanced Classification, value for Response [1]: {targets.value_counts()[1] / targets.value_counts()[0]*100:0.2f}%"
)
features = df.drop("response", axis=1).values

kfold = StratifiedKFold(n_splits=5, shuffle=True)

# enumerate the splits
for train_ix, test_ix in kfold.split(features, targets):
    x_train, x_test = features[train_ix], features[test_ix]
    y_train, y_test = targets[train_ix], targets[test_ix]

    # makes sure its even for all k-folds
    train_0, train_1 = len(y_train[y_train == 0]), len(y_train[y_train == 1])
    test_0, test_1 = len(y_test[y_test == 0]), len(y_test[y_test == 1])
    print(f"Train: 0={train_0}, 1={train_1}, Test: 0={test_0}, 1={test_1}")
