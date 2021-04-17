import logging
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

def get_train(*args):
    """Get training dataset for KDD 10 percent"""
    return _get_dataset()['x_train'],_get_dataset()['y_train']

def get_test(*args):
    """Get testing dataset for KDD 10 percent"""
    return _get_dataset()['x_test'],_get_dataset()['y_test']

def get_shape_input():
    """Get shape of the dataset for KDD 10 percent"""
    return (None, 29)

def get_shape_label():
    """Get shape of the labels in KDD 10 percent"""
    return (None,)

def _get_dataset():
    """ Gets the basic dataset
    Returns :
            dataset (dict): containing the data
                dataset['x_train'] (np.array): training images shape
                (?, 120)
                dataset['y_train'] (np.array): training labels shape
                (?,)
                dataset['x_test'] (np.array): testing images shape
                (?, 120)
                dataset['y_test'] (np.array): testing labels shape
                (?,)
    """
    js = '/Users/oguzkaplan/Documents/repo/thesis/cc_fraud_data/test_cc.csv'
    df_test = pd.read_csv(js)
    js = '/Users/oguzkaplan/Documents/repo/thesis/cc_fraud_data/train_cc.csv'
    df_train = pd.read_csv(js)

    df_train = df_train.rename(columns = {'Class':'label'})
    df_test = df_test.rename(columns = {'Class':'label'})

    df_train.label = 1
    df_test.label = np.abs(df_test.label-1)

    df_test = df_test.rename(columns={'Unnamed: 0': 'id'}).set_index('id')
    df_train = df_train.rename(columns={'Unnamed: 0': 'id'}).set_index('id')

    df_train.Amount = df_train.Amount.replace([-np.inf,np.inf],0)
    df_test.Amount = df_test.Amount.replace([-np.inf,np.inf],0)

    # df_test_y = df_test.Class
    # df_test = df_test.drop('Class',axis = 1)
    # df_train = df_train.drop('Class',axis = 1)
  

    x_train, y_train = _to_xy(df_train, target='label')
    y_train = y_train.flatten().astype(int)
    x_test, y_test = _to_xy(df_test, target='label')
    y_test = y_test.flatten().astype(int)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print(x_test.shape)



    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    return dataset


def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)

def _col_names():
    """Column names of the dataframe"""
    return ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount','label']
