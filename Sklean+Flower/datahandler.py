import numpy as np
import pandas as pd # use version==1.2.5 incase you want to run pandas profiling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

data_path = "data/kddcup_data"

# Data is available at: https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
# description of column names at: https://kdd.ics.uci.edu/databases/kddcup99/kddcup.names

col_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate']

num_col = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
                        'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'count',
                        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']


def data_processor():
    df = pd.read_csv(data_path, names=col_names+["threat_type"]) # threat type is the target
    # print(df.head())

    # get the list of threat types and their total count
    # print(f'The total number of unique threats in the data is... : ', len(df['threat_type'].unique()))
    # print(' ')
    # print(f'The unique data threat types are... : ', df['threat_type'].unique())

    # do some preprocessing
    # print(' ')
    df['threat_type'] = df['threat_type'].str.replace('.', '', regex=True)
    # df = df.drop_duplicates()
    
    #drop the columns whose total count is less than 20
    indexNames = df[(df['threat_type'] == 'spy') | (df['threat_type'] == 'perl') | (df['threat_type'] == 'phf') 
                | (df['threat_type'] == 'multihop') | (df['threat_type'] == 'ftp_write') | (df['threat_type'] == 'loadmodule') 
                | (df['threat_type'] == 'rootkit') | (df['threat_type'] == 'imap')].index
    df.drop(indexNames , inplace=True)

    # 34 numerical columns are considered for training
    num_df = df[num_col]

    # Lets remove the numerical columns with constant value
    X = num_df.loc[:, (num_df != num_df.iloc[0]).any()].values
    # print(' ')
    # print(f'The shape of the feature vector is... : ',X.shape)
    #scale the num_df 


    # labelencode the target variable
    threat_types = df["threat_type"].values
    encoder = LabelEncoder()
    # encoder = OneHotEncoder()
    # use LabelEncoder to encode the threat types in numeric values
    y = encoder.fit_transform(threat_types)
    # print(' ')
    # print("Shape of target vector is... : ",y.shape)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=13, stratify=y)

    scaler = StandardScaler()  
    a = scaler.fit(X_train)
    X_train = a.transform(X_train)
    X_test = scaler.transform(X_test)

    # print(y_test)
    print(len(np.unique(y_train)), len(np.unique(y_test)))
    # unique, counts = np.unique(y_train, return_counts=True)
    # unique1, counts1 = np.unique(y_test, return_counts=True)

    # print(np.array(np.unique(y_train, return_counts=True)).T)
    # print(np.array(np.unique(y_test, return_counts=True)).T)

    # print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test))
    # print(np.unique(y_test), np.unique(y_train))

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = data_processor()