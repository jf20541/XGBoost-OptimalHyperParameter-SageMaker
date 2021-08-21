# XGBoost-OptimalHyperParameters-AWS-SageMaker

## Objective
Build a Supervised Learning Classification model to predict whether the customers will be interested in vehicle insurance provided by the company. With an imbalanced binary classification dataset (~10% of target values are interested). The primary metric to evaluated XGBoost Model will be ROC-AUC, use Bayesian Optimization Gaussian Process to optimize hyper-parameters, and use Stratified K-Fold to reduce overfitting.

## Model and Metric 
XGBoost Classifier: An ensembled classification model by combining the outputs from individual trees using boosting. It combines weak learners sequentially so that each tree corrects the residuals from the previous trees. Trees are added until no further improvements can be made to the model.

Metric: Receiver Operating Characteristic Curve (ROC AUC) and Minimize ```binary:logistic:``` XGBoost loss function for binary classification.

## Repository File Structure
    ├── src          
    │   ├── main.py                 # Initiated the XGBoost Classifier and optimized its parameter with Bayesian Optimization
    │   ├── data.py                 # Cleaned and featured engineered the dataset
    │   ├── create_folds.py         # Stratified K-Fold cross-validation with an imbalanced dataset
    │   └── config.py               # Define path as global variable
    ├── sagemaker
    │   └── xgboost_sagemaker.ipynb # Fit the XGBoost with defined optimal Hyper-Parameters using AWS SageMaker
    ├── inputs
    │   ├── train.csv               # Training dataset
    │   ├── test.csv                # Testing dataset
    │   └── clean_data.csv          # Cleaned data 
    ├── notebooks
    │   └── healthinsurance.ipynb   # Exploratory Data Analysis and Feature Engineering
    ├── requierments.txt            # Packages used for project
    └── README.md
    
## Output
```bash
XGBoost Classifier with Bayesian Optimization Gaussian: 0.9085 ROC-AUC 

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
## Parameters
- ```max_depth:``` Maximum depth of a tree
- ```gamma:``` Minimum loss reduction required to make a further partition on a leaf node of the tree
- ```reg_alpha:``` L1 regularization term on weights
- ```reg_lambda:``` L2 regularization term on weights
- ```min_child_weight:``` Minimum sum of instance weight (hessian) needed in a child
- ```eta:``` Learning rate, step size shrinkage used in update to prevents overfitting.
- ```colsample_bytree:``` Subsample ratio of columns when constructing each tree
- ```base_score:``` Initial prediction score of all instances, global bias. Since dataset is imbalanced (~90% would be a recommended base score)

## XGBoost Step-by-Step

### Summary 
Calculate the Similarity Score and Gain to determine how to split the data and we prune the tree by calculating the difference between Gain values and Gamma (hyper-parameter). Then calculate the output values for the leaves. Define a regularization Lambda parameter which helps reduce the similarity score and smaller outputs values for the leaves. 
1. XGBoost makes an initial prediction as a probability to a log(odds) value for classification
2. Defines a thresholds to clusters the residuals
3. Calculates the similarity scores by fitting the trees to the residuals and gets the prediction from the previous tree
4. Set Lambda as the regularization parameter (reduces Similarity Scores thus reduce Gain value)
5. Calculate the Gain by adding Similarity Scores from all leafs to maximize the Gain value to set a proper threshold
6. Define Gamma, calculates the difference between the Gain Value associciated with the lowest branch if g<0 the leaf is pruned.
7. When tree is finished, we add the log(odds)-previous prediction to the output of the tree multiplied by the Learning Rate (eta) 


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
