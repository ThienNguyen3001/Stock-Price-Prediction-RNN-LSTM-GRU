import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
np.random.seed(38)
scaler = MinMaxScaler(feature_range=(-1, 1))
def split_data(prices, window):
    raw = prices.to_numpy()
    data = []
    plotting = []
    
    for index in range(len(raw) - window): 
        plotting.append(raw[index][0])
        data.append(raw[index: index + window])
    
    plotting = pd.DataFrame(plotting)
    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    X_train = data[:train_set_size,:-1,]
    X_train = scaler.fit_transform(X_train.reshape(-1,1))
    X_train = X_train.reshape(train_set_size, window - 1, 1)
    y_train = data[:train_set_size,-1,:]
    y_train = scaler.fit_transform(y_train)
    
    X_test = data[:test_set_size,:-1,]
    X_test = scaler.fit_transform(X_test.reshape(-1,1))
    X_test = X_test.reshape(test_set_size, window - 1, 1)
    y_test = data[:test_set_size,-1,]
    y_test = scaler.fit_transform(y_test)
    
    return [scaler,X_train, y_train, X_test, y_test]
