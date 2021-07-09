import pandas as pd 
import config
from sklearn.preprocessing import LabelEncoder

def clean_colname(df):
    #lower case columns, no spaces, no dashes
    df.columns = [x.lower().replace(" ", "_").replace("-","_").replace(".","_") for x in df.columns]
    return df.columns


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
         # array of column names to encode
        self.columns = columns

    def fit(self,X,y=None):
        return self 

    def transform(self,X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    
if __name__ == '__main__':
    test = pd.read_csv(config.TESTING_FILE)
    sample = pd.read_csv(config.SAMPLING_FILE)
    train = pd.read_csv(config.TRAINING_FILE)
    
    test = test.set_index('id').join(sample.set_index('id'))
    train = train.set_index('id')
    df = pd.concat([train, test])
    col = clean_colname(df)
    df.columns = col
    df = MultiColumnLabelEncoder(columns = ['gender', 'vehicle_age', 'vehicle_damage']).fit_transform(df)
    if df.isnull().sum().any() == False:
        print('Data is Clean, No Null Values Found')
        df.to_csv(config.CLEAN_FILE, index=False)
    else:
        print('Found Null Values')
    
# https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn