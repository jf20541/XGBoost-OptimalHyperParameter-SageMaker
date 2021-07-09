import pandas as pd
import numpy as np 
import config
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from functools import partial
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from skopt import gp_minimize
from skopt import space
from skopt import BayesSearchCV





model = xgb.XGBClassifier()
model.fit(x_train, y_train)
# make predictions for test data
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))






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

    # initiate RandomForestClassifie and K-fold (5)
    model = xgb.XGBClassifier(**params)
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


if __name__ == "__main__":
    df = pd.read_csv(config.CLEAN_FILE)
    targets = df['response'].values
    features = df.drop('response', axis=1).values

    # define the range of input values to test the Bayes_op to create prop-distribution
    param_space = [
        space.Categorical(["binary:logistic"], name="objective"),
        space.Integer(2, 20, name="max_depth"),
        space.Integer(2, 20, name="min_child_weight"),
        space.Integer(10, 1000, name="n_estimators"),
        space.Real(1e-3, 100, "log-uniform", name="learning_rate"),
        space.Real(1e-2, 100, "log-uniform", name="eta"),
        space.Real(0.05, 0.8, name="gamma"),
        space.Real(0.1, 0.9, name="subsample"),
        space.Real(0.5, 1, name="colsample_bytree")
    ]

    
    
    param_names = [
        "objective",
        "max_depth",
        "min_child_weight",
        "n_estimators",
        "learning_rate",
        "eta",
        "gamma",
        "subsample",
        "colsample_bytree" 
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
    
    
    
# def objective(space):
#     clf=xgb.XGBClassifier(
#                     n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
#                     reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
#                     colsample_bytree=int(space['colsample_bytree']))
    
#     evaluation = [( X_train, y_train), ( X_test, y_test)]
    
#     clf.fit(X_train, y_train,
#             eval_set=evaluation, eval_metric="auc",
#             early_stopping_rounds=10,verbose=False)
    

#     pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, pred>0.5)
#     print ("SCORE:", accuracy)
#     return {'loss': -accuracy, 'status': STATUS_OK }



y_score = xgb_model.predict_proba(x_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)

title('XGBoost ROC curve')
xlabel('FPR (Precision)')
ylabel('TPR (Recall)')

plot(fpr,tpr)
plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))



from catboost import CatBoostClassifier

cat = CatBoostClassifier(learning_rate=0.03, l2_leaf_reg=1, iterations= 500, depth= 9, border_count= 20,eval_metric = 'AUC')

cat= cat.fit(X_train, y_train,cat_features=cat_col,eval_set=(X_test, y_test),early_stopping_rounds=70,verbose=50)

pred_proba = cat.predict_proba(X_test)[:, 1]
print('CatBoost ROC AUC SCORE: {}'.format(roc_auc_score(y_test,pred_proba)))



# model = xgb.XGBClassifier()
# model.fit(x_train, y_train)
# # make predictions for test data
# y_pred = model.predict(x_test)
# predictions = [round(value) for value in y_pred]
# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))



# kfold = StratifiedKFold(n_splits=5, shuffle=True)

# # enumerate the splits and summarize the distributions
# for train_ix, test_ix in kfold.split(features, targets):
# 	x_train, x_test = features[train_ix], features[test_ix]
# 	y_train, y_test = targets[train_ix], targets[test_ix]
# 	# train_0, train_1 = len(y_train[y_train==0]), len(y_train[y_train==1])
# 	# test_0, test_1 = len(y_test[y_test==0]), len(y_test[y_test==1])
# 	# print(f'Train: 0={train_0}, 1={train_1}, Test: 0={test_0}, 1={test_1}')
 
#     d_train = xgb.DMatrix(x_train, y_train)
#     d_valid = xgb.DMatrix(x_test, y_test)
#     # watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#     evaluation = [( x_train, y_train), ( x_test, y_test)]