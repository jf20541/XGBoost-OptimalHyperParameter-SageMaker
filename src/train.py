from sklearn.model_selection import StratifiedKFold
import config
import pandas as pd
from sklearn.metrics import roc_auc_score
import xgboost as xgb


df = pd.read_csv(config.CLEAN_FILE)
targets = df['response'].values
features =  df.drop('response', axis=1).values

model = xgb.XGBClassifier(objective='binary:logistic', num_classes=2, use_label_encoder=False)
fold = StratifiedKFold(n_splits=5, shuffle=True)
pred = []
score =[]
for train_idx , test_idx in fold.split(features, targets):
    x_train, x_test, y_train, y_test = features[train_idx], features[test_idx], targets[train_idx], targets[test_idx]
    lgb= model.fit(x_train, y_train)
    pred_proba = model.predict(x_test)
    score.append(roc_auc_score(y_test, pred_proba))
    print(score)