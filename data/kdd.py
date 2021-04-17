import logging
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

logger = logging.getLogger(__name__)

def get_train(*args):
    """Get training dataset for KDD 10 percent"""
    return _get_dataset()['x_train'],_get_dataset()['y_train']

def get_test(*args):
    """Get testing dataset for KDD 10 percent"""
    return _get_dataset()['x_test'],_get_dataset()['y_test']

def get_shape_input():
    """Get shape of the dataset for KDD 10 percent"""
    return (None, 39)

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


    '''
    smurf.              280790
    neptune.            107201
    normal.              97277
    back.                 2203 kotu 15 epoch
    satan.                1589 cok az ok 15 epoch
    ipsweep.              1247 ok 15 epoch anogan daha iyi
    portsweep.            1040 ok 15 epoch  biz daha iyiyiz
    warezclient.          1020 ok 15 epoch  anogan daha iyi
    teardrop.              979 ok 15 epoch
    pod.                   264 bunlarin birlesiminde biz daha iyiyiz 20 epoch
    nmap.                  231 bunlarin birlesiminde biz daha iyiyiz 20 epoch
    guess_passwd.           53 bunlarin birlesiminde biz daha iyiyiz 20 epoch
    buffer_overflow.        30 bunlarin birlesiminde biz daha iyiyiz 20 epoch
    land.                   21 bunlarin birlesiminde biz daha iyiyiz 20 epoch
    warezmaster.            20 bunlarin birlesiminde biz daha iyiyiz 20 epoch
    '''

    to_be_detected = ['pod.','nmap.','guess_passwd.','buffer_overflow.','land.','warezmaster.']
    # to_be_detected = ['back.']#ipsweep.,portsweep.,warezclient.,teardrop.
                             # satan.,back. kotu


    col_names = _col_names()
    df = pd.read_csv("data/kddcup.data_10_percent_corrected", header=None, names=col_names)
    # text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
    
    #drop_cols = ['protocol_type', 'service', 'flag', 'land']
    drop_cols = []
    text_l = ['logged_in', 'is_host_login', 'is_guest_login','protocol_type', 'service', 'flag', 'land']

    for name in text_l:
        _encode_text_dummy(df, name)

    df.drop(drop_cols,axis=1,inplace=True)

    number = df[df.label.isin(to_be_detected)].shape[0]
    idx = df[df['label']=='normal.'].sample(97277-number).index
    train = df.loc[idx]
    idx = idx.to_list()

    # number = df[(df['label'].index.isin(idx)==False) & (df['label']=='smurf.')].shape[0]
    
    test_idx = df[(df.index.isin(idx)==False) & (df['label']=='normal.')].sample(number).index 
    # all_normal = []
    # all_normal.extend(test_idx.to_list())
    # all_normal.extend(idx)
    
    if number > df.label.value_counts().loc[to_be_detected].sum() :
        number = df.label.value_counts().loc[to_be_detected].sum()
        rest_test = df[df['label'].isin(to_be_detected)].sample(number).index
    else :
        rest_test = df[df['label'].isin(to_be_detected)].sample(number).index

    test_idx = test_idx.to_list()
    test_idx.extend(rest_test.to_list())

    test = df.loc[test_idx]
    #print('##################',test.shape,'####################')
    #print(test.label.value_counts())


    labels = train['label'].copy()
    labels[labels != 'normal.'] = 0
    labels[labels == 'normal.'] = 1
    train['label'] = labels

    labels = test['label'].copy()
    labels[labels != 'normal.'] = 0
    labels[labels == 'normal.'] = 1
    test['label'] = labels

    # df_train = df.sample(frac=0.5, random_state=42)
    # df_test = df.loc[~df.index.isin(df_train.index)]

    df_train = train.copy()
    df_test = test.copy()

    #df_train.to_csv('df_train_121.csv')
    #df_test.to_csv('df_test_121.csv')
    
    df_train = pd.read_csv('df_train.csv').drop('Unnamed: 0',axis =1)
    df_test = pd.read_csv('df_test.csv').drop('Unnamed: 0',axis =1)

    print('##################',df_test.shape,'####################')
    print(df_test.label.value_counts())
    
    x_train, y_train = _to_xy(df_train, target='label')
    y_train = y_train.flatten().astype(int)
    x_test, y_test = _to_xy(df_test, target='label')
    y_test = y_test.flatten().astype(int)

    # x_train = x_train[y_train != 1]
    # y_train = y_train[y_train != 1]

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    return dataset

# def _get_adapted_dataset(split):
#     """ Gets the adapted dataset for the experiments

#     Args :
#             split (str): train or test
#     Returns :
#             (tuple): <training, testing> images and labels
#     """
#     dataset = _get_dataset()
#     key_img = 'x_' + split
#     key_lbl = 'y_' + split

#     if split != 'train':
#         dataset[key_img], dataset[key_lbl] = _adapt(dataset[key_img],
#                                                     dataset[key_lbl])

#     return (dataset[key_img], dataset[key_lbl])

def _encode_text_dummy(df, name):
    """Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1]
    for red,green,blue)
    """
    dummies = pd.get_dummies(df.loc[:,name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

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
    return ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

def _adapt(x, y, rho=0.2):
    """Adapt the ratio of normal/anomalous data"""

    # Normal data: label =0, anomalous data: label =1

    rng = np.random.RandomState(42) # seed shuffling

    inliersx = x[y == 0]
    inliersy = y[y == 0]
    outliersx = x[y == 1]
    outliersy = y[y == 1]

    size_outliers = outliersx.shape[0]
    inds = rng.permutation(size_outliers)
    outliersx, outliersy = outliersx[inds], outliersy[inds]

    size_test = inliersx.shape[0]
    out_size_test = int(size_test*rho/(1-rho))

    outestx = outliersx[:out_size_test]
    outesty = outliersy[:out_size_test]

    testx = np.concatenate((inliersx,outestx), axis=0)
    testy = np.concatenate((inliersy,outesty), axis=0)

    size_test = testx.shape[0]
    inds = rng.permutation(size_test)
    testx, testy = testx[inds], testy[inds]

    return testx, testy