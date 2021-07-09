# example of stratified k-fold cross-validation with an imbalanced dataset
from sklearn.model_selection import StratifiedKFold
import pandas as pd 
import config
from pprint import pprint 
import xgboost as xgb


df = pd.read_csv(config.CLEAN_FILE)
targets = df['response'].values
features = df.drop('response', axis=1).values

kfold = StratifiedKFold(n_splits=5, shuffle=True)

# enumerate the splits and summarize the distributions
for train_ix, test_ix in kfold.split(features, targets):
	x_train, x_test = features[train_ix], features[test_ix]
	y_train, y_test = targets[train_ix], targets[test_ix]


model = xgb.XGBClassifier(n_estimators=50, max_depth=3, objective='binary:logistic', use_label_encoder=False)

model.fit(x_train, y_train)
predict = model.predict(x_test)
from sklearn.metrics import precision_score, recall_score, accuracy_score


# print metrics for Test set

print("Precision = {}".format(precision_score(y_test, predict)))
print("Recall = {}".format(recall_score(y_test, predict)))
print("Accuracy = {}".format(accuracy_score(y_test, predict)))