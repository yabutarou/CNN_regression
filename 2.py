import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from tensorflow.keras.layers import Activation, Conv2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
import os


#設定
#慣習として２のｎ乗
batch_size = 16
epochs = 20

#画像ディレクトリ
test_dir ="cifar230120_high_and_low"
folder = os.listdir(test_dir)
#元画像640x480をresizeする際の大きさ設定
x_image_size = 28#640→200
y_image_size = 28#480→200
dense_size  = len(folder)


X = []
Y = []
for index, name in enumerate(folder):
    dir = test_dir + "/*" + name
    files = glob.glob(dir + "/*.jpg")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((x_image_size, y_image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)
#0~255で表される画像データを正規化で0~1で表せるようにする
X = np.array(X)
Y = np.array(Y)
X = X.astype('float32')
X = X / 255.0

Y = np_utils.to_categorical(Y, dense_size)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)


model = Sequential()
#keras.layers.Dense((x_image_size*y_image_size),activation='relu',kernel_initializer = tf.initializers.he_normal())#ReLUを用いるときはHeの初期値を使うとよい
model.add(Conv2D(32, (3, 3), kernel_initializer = "he_uniform", padding='same',input_shape=X_train.shape[1:]))#1段目の畳み込みフィルターの個数を指定 #出力は32個の画像になる#X_train.shape[1:] = (28,28) 画像の大きさ[1:]１以上#(3,3)=カーネルサイズ#32=カーネル枚数
#model.add(BatchNormalization())
model.add(Activation('relu'))#活性化関数#複数の入力の総和が条件を満たしたとき1を出力する
model.add(Conv2D(32, (3, 3), kernel_initializer = "he_uniform"))#"Heの初期化"Conv2D(32, (3, 3), kernel_initializer = "he_uniform")#活性化関数がReLUの時はHeの初期化を用いると良い。
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))#プーリング層#(2,2)サイズ
model.add(Dropout(0.25))#全結合層とのつながりを25%無効

model.add(Conv2D(64, (3, 3), kernel_initializer = "he_uniform", padding='same'))#2段目の畳み込みフィルターの個数を指定#今回は入力レイヤ32出力レイヤ64
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), kernel_initializer = "he_uniform"))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())#４次元から1次元に変換
model.add(Dense(512))#出力が512
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(dense_size))
#model.add(BatchNormalization())
model.add(Activation('softmax'))#確率に変換

model.summary()


optimizers ="Adadelta"
results = {}

model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=['accuracy'])
results[0]= model.fit(X_train, y_train, validation_split=0.2, epochs=epochs,batch_size=batch_size)

model_json_str = model.to_json()
open('model.json', 'w').write(model_json_str)
model.save('weights230120_test_high_and_low_batch16_epoch20.h5');




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