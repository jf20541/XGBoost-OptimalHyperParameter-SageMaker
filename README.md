# XGBoostClassifierHyperParameter


## Objective
Build a Supervised Learning Classification model to predict whether the customers will be interested in vehicle insurance provided by the company. With an imbalanced binary classification dataset (~10% of target values are interested). The primary metric to evaluated XGBoost Model will be ROC-AUC, use Bayesian Optimization Gaussian Process to optimize hyper-parameters, and use Stratified K-Fold to reduce overfitting.


## Model
XGBoost Classifier: An ensembled classification model by combining the outputs from individual trees using boosting. It combines weak learners sequentially so that each tree corrects the residuals from the previous trees. Trees are added until no further improvements can be made to the model.


## Parameters
- max_depth: Maximum depth of a tree
- gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree
- reg_alpha: L1 regularization term on weights
- reg_lambda: L2 regularization term on weights
- min_child_weight: Minimum sum of instance weight (hessian) needed in a child
- eta: Learning rate, step size shrinkage used in update to prevents overfitting.
- colsample_bytree: Subsample ratio of columns when constructing each tree
- base_score: Initial prediction score of all instances, global bias. Since dataset is imbalanced (~90% would be a recommended base score)


## Metric
ROC-Area Under Curve (ROC AUC) 

## Output
```bash
XGBoost Classifier with Bayesian Optimization Gaussian: 90.85%

Optimal Hyper-Parameters:

'base_score': 0.5074025941621864
'colsample_bytree': 0.7805503460343888
'gamma': 8.815330570605372 
'max_depth': 7.0
'min_child_weight': 1.0
'reg_alpha': 170.0
'reg_lambda': 0.8244076191380255
'eta': 0.24528128897339257
```


### Code
Created 5 modules
- `main.py`: 
- `data.py`: 
- `create_folds.py`: initiate StratefiedKFold since target values are skewed distribution.
- `config.py`: Defined file paths as global variable


## Data
[Kaggle Dataset](https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction)
```bash
Target (binary)
Response               int64

Features: 
Gender                 int64
Age                    int64
Driving_License        int64
Region_Code          float64
Previously_Insured     int64
Vehicle_Age            int64
Vehicle_Damage         int64
Annual_Premium       float64
PolicySalesChannel   float64
Vintage              float64
```
## Sources
https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction
https://towardsdatascience.com/beginners-guide-to-xgboost-for-classification-problems-50f75aac5390
https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
