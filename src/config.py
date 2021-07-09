TRAINING_FILE='../inputs/train.csv'
TESTING_FILE='../inputs/test.csv'
SAMPLING_FILE='../inputs/sample_submission.csv'
CLEAN_FILE='../inputs/clean_data.csv'

# stratified k-fold cross-validation 
# If targets are skewed, always use AUC/ROC

"""
If the distribution shows that some classes have alot and some dont, we CANT do simple KFold since
we dont habe equal distribution of targets in every fold. CAN'T BE USED FOR REGRESSION PROBLEMS unless 
the distribution of targets is not consistent. You first need to divide the target into bins using the
Sturge’s Rule to calculate the number of bins

$Number of Bins = 1 + log2(N)$
and then we can use stratified k-fold in the same way as for classification problems
"""

# Fix Cross-Validation for Imbalanced Classification
# Specifically, we can split a dataset randomly, although in such a way that maintains the same class distribution in each subset. 
# This is called stratification or stratified sampling and the target variable (y), the class, is used to control the sampling process.

# MinMaxScaler