#CNN+LSTM
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from tensorflow.keras.layers import Activation, Conv2D, Flatten, Dense, Dropout, BatchNormalization
from keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
import os

def create_dataset(ds, look_back=1):
    X, y = [], []
    for i in range(len(ds)-look_back):
        X.append(ds[i : i+look_back])#学習の学習例）[0:3]=0,1,2
        y.append(ds[i+look_back])#学習の推測例）[3]=3
    X = np.reshape(np.array(X), [-1, look_back, 1])#np.array=numpy配列にする#3次元の縦ベクトル（縦一列）
    y = np.reshape(np.array(y), [-1, 1])
    return X, y#X=学習(学習) y=推論(学習)

def create_dataset2(ds, look_back=1):
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
    x_mean = x.mean(axis=axis, keepdims=True)#平均
    x_std = x.std(axis=axis, keepdims=True, ddof=ddof)#標準偏差
    return (x - x_mean) / x_std , x_mean, x_std
#設定
#慣習として２のｎ乗
batch_size = 16
epochs = 20

#画像ディレクトリ
test_dir ="Datasets_kaiki_20230224"
folder = os.listdir(test_dir)
#元画像640x480をresizeする際の大きさ設定
x_image_size = 28#640→200#横
y_image_size = 28#480→200#縦
dense_size  = len(folder)
look_back = 5

X = []
Y = []
for index, name in enumerate(folder):
    dir = test_dir + "/*" + name
    files = glob.glob(dir + "/*.jpg")
    for i, file in enumerate(files):
        print(file)
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((x_image_size, y_image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(name)

#0~255で表される画像データを正規化で0~1で表せるようにする
X = np.array(X)#画像
Y = np.array(Y)#正解ラベル
X = X.astype('float32')
X = X / 255.0 
print(X.shape)
print(Y.shape)
#Y = np_utils.to_categorical(Y, dense_size)#回帰の時必要ないかも
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, shuffle=False)
#print(y_train)
#print(y_test)
model = Sequential()
#keras.layers.Dense((x_image_size*y_image_size),activation='relu',kernel_initializer = tf.initializers.he_normal())#ReLUを用いるときはHeの初期値を使うとよい
model.add(Conv2D(32, (3, 3), kernel_initializer = "he_uniform", padding='same',input_shape=X_train.shape[1:]))#1段目の畳み込みフィルターの個数を指定 #出力は32個の画像になる#X_train.shape[1:] = (28,28) 画像の大きさ[1:]１以上#(3,3)=カーネルサイズ#32=カーネル枚数
#model.add(Activation('relu'))#活性化関数#複数の入力の総和が条件を満たしたとき1を出力する
#model.add(Conv2D(32, (3, 3), kernel_initializer = "he_uniform"))#"Heの初期化"Conv2D(32, (3, 3), kernel_initializer = "he_uniform")#活性化関数がReLUの時はHeの初期化を用いると良い。
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))#プーリング層#(2,2)サイズ
#model.add(Dropout(0.25))#全結合層とのつながりを25%無効
#model.add(Conv2D(64, (3, 3), kernel_initializer = "he_uniform", padding='same'))#2段目の畳み込みフィルターの個数を指定#今回は入力レイヤ32出力レイヤ64
#model.add(Activation('relu'))
#model.add(Conv2D(64, (3, 3), kernel_initializer = "he_uniform"))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())#４次元から1次元に変換,空間的な情報が消える

ds = model.predict(X_train) #(訓練画像枚数（テストデータ以外）,次元)
ds = np.array(ds)

#正規化
v_min = np.min(np.abs(ds))#v_min = 0
v_max = np.max(np.abs(ds))#v_max = 90(今回は)
ds -= v_min
ds /= v_max - v_min
ds = np.array(ds)
#print(ds.shape)
X_LSTM, y_LSTM = create_dataset2(ds, look_back)
print(X_LSTM.shape)
X_train_LSTM, y_train_LSTM, X_test_LSTM, y_test_LSTM = split_data(X_LSTM, y_LSTM,0.15)#テストデータあり
#X_train_LSTM, y_train_LSTM, X_test_LSTM, y_test_LSTM = train_test_split(X_LSTM, y_LSTM, test_size=0.15, shuffle=False)
print(X_train_LSTM.shape)
print(y_train_LSTM.shape)
model.add(LSTM(units=50,return_sequences=True,input_shape=(look_back, ds.shape[1])))#return_sequences=TrueはLSTMの全ての出力系列が次のレイヤーに入力される。LSTMを多段に積むときの必須指定で最終層以外はTrue
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(1, activation='linear')) # ← ここが出力層！
model.summary()
optimizers ="Adadelta"
results = {}

model.compile(loss='mean_squared_error', optimizer=optimizers, metrics=['accuracy'])
results[0]= model.fit(X_train_LSTM, y_train_LSTM, validation_split=0.2, epochs=epochs,batch_size=batch_size)

model_json_str = model.to_json()
open('model.json', 'w').write(model_json_str)
model.save('CNN_LSTM_kaiki_batch16_epoch20.h5');




#グラフ化
history = model.fit(
    X_train,
    y_train,
    batch_size,
    epochs,
    verbose=1,
    validation_data=(X_test, y_test),
)

metrics = ['loss', 'accuracy']  # 使用する評価関数を指定

plt.figure(figsize=(10, 5))  # グラフを表示するスペースを用意

for i in range(len(metrics)):

    metric = metrics[i]

    plt.subplot(1, 2, i+1)  # figureを1×2のスペースに分け、i+1番目のスペースを使う
    plt.title(metric)  # グラフのタイトルを表示
    
    plt_train = history.history[metric]  # historyから訓練データの評価を取り出す
    plt_test = history.history['val_' + metric]  # historyからテストデータの評価を取り出す
    
    plt.plot(plt_train, label='training')  # 訓練データの評価をグラフにプロット
    plt.plot(plt_test, label='test')  # テストデータの評価をグラフにプロット
    plt.legend()  # ラベルの表示
    
plt.show()  # グラフの表示

#####################################################################################################################################

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras.utils import to_categorical

# データの前処理
x_train = ...
y_train = to_categorical(...)
x_test = ...
y_test = to_categorical(...)

# model.loadで代用########################################################################################################
model_cnn = Sequential()
model_cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Dropout(0.25))
model_cnn.add(Flatten())
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(num_classes, activation='softmax'))###########################################ここも回帰問題の出力にするべき？
model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
################################################################################################################################



# softmaxを使わずに特徴抽出だけ行う#######################################################################################################
model_feat = Sequential()
model_feat.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model_feat.add(MaxPooling2D(pool_size=(2, 2)))
model_feat.add(Dropout(0.25))
model_feat.add(Flatten())
model_feat.add(Dense(128, activation='relu'))
model_feat.add(Dropout(0.5))
model_feat.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
feat_train = model_feat.predict(x_train)
feat_test = model_feat.predict(x_test)

# LSTMモデルの定義と学習
model_lstm = Sequential()
model_lstm.add(LSTM(64, input_shape=(time_steps, feature_dim)))#feature_dimは今回は128 最後の全結合層の次元
model_lstm.add(Dense(1, activation='linear')) 
#model_lstm.add(Dense(num_classes, activation='softmax'))
model_lstm.compile(loss='mean_squared_error', optimizer=optimizers, metrics=['accuracy'])
#model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.fit(feat_train, y_train, batch_size=32, epochs=10, validation_data=(feat_test, y_test))
