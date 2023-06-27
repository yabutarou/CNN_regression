import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def create_dataset(ds, look_back=1):
    X, y = [], []
    for i in range(len(ds)-look_back):
        X.append(ds[i : i+look_back])#学習の学習例）[0:3]=0,1,2
        y.append(ds[i+look_back])#学習の推測例）[3]=3
    X = np.reshape(np.array(X), [-1, look_back, 1])#np.array=numpy配列にする#3次元の縦ベクトル（縦一列）
    y = np.reshape(np.array(y), [-1, 1])
    return X, y#X=学習(学習) y=推論(学習)

def split_data(X, y, test_size=0.1):
    pos = int(round(len(X) * (1-test_size)))
    X_train, y_train = X[:pos], y[:pos]
    X_test, y_test = X[pos:], y[pos:]
    return X_train, y_train, X_test, y_test


def standardization(x, axis=None, ddof=0):
    x_mean = x.mean(axis=axis, keepdims=True)
    x_std = x.std(axis=axis, keepdims=True, ddof=ddof)
    return (x - x_mean) / x_std

############################################################################データセット#####################################################################
df = pd.read_csv("LSTM_IAS_input2.csv",usecols=[1],header=None,skiprows=1,skipfooter=3,engine="python")#skiprows=先頭何行スキップするか、skipfooter=行末何行スキップするか、
ds = df.filter(['A', 'B'])#''の中の題名から始まるデータを全て学習する
ds = df.values.astype("float32")
print("ds.shape",ds.shape)
data = []
for i in range(ds.shape[0]-1):#(ds.shape[0]-1)=全データ数 - 1
    #data.append(ds[i+1]-ds[i])
    data.append(ds[i])
data = np.array(data)#前回値との差をとったもの＝data
#正規化
v_min = np.min(np.abs(data))#v_min = 0
v_max = np.max(np.abs(data))#v_max = 90(今回は)
data -= v_min
data /= v_max - v_min
#標準化
#data_standard = standardization(data)
plt.figure(figsize=(8,8))
plt.plot(data)
plt.show()
plt.clf()
look_back = 5
X, y = create_dataset(data, look_back)
#print("X:{},y:{}".format(X.shape, y.shape))
X_train, y_train, X_test, y_test = split_data(X, y, 0)#テストデータに分割（モデルの精度評価のため）
#print("X_train:{},y_train:{},X_test:{},y:test{}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
############################################################################データセット#####################################################################

###############################################################################学習##########################################################################
model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, batch_size=32, epochs=2)
model.summary()
