import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
from PIL import Image
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers import ConvLSTM2D
from keras.layers import Activation,BatchNormalization, Flatten,Dense, Dropout
import numpy as np
import pylab as plt
import os
from natsort import natsorted
test_dir ="221109_kirinuki_20230329_dir"
folder = os.listdir(test_dir)
#元画像640x480をresizeする際の大きさ設定
x_image_size = 28#640→200#横
y_image_size = 28#480→200#縦
dense_size  = len(folder)

df = pd.read_csv("LSTM_IAS_Datasets.csv",usecols=[1],header=None,skiprows=1,skipfooter=3,engine="python")#skiprows=先頭何行スキップするか、skipfooter=行末何行スキップするか、
ds = df.values.astype("float32")
data = []
for i in range(ds.shape[0]-1):
    data.append(ds[i])
Y = np.array(data)
imgs = np.empty((0, y_image_size, x_image_size, 3))


files = natsorted([os.path.basename(p) for p in glob.glob(test_dir + "/**") if os.path.isfile(p)])
for i, file in enumerate(files):
    path = test_dir + "/" + file
    image = Image.open(path)
    image = image.convert("RGB")
    image = image.resize((x_image_size, y_image_size))
    img_np = np.array(image).reshape(1, x_image_size, x_image_size, 3)
    imgs = np.append(imgs, img_np, axis=0)

# 時系列で学習できる形式に整える
n_seq = 3
n_sample = imgs.shape[0] - n_seq
print(n_sample)

x = np.zeros((n_sample, n_seq, y_image_size, x_image_size, 3))
y = np.zeros((dense_size-n_seq))
for i in range(n_sample):
    x[i] = imgs[i:i+n_seq]
    y[i] = Y[i+n_seq]
print(len(y))
print(len(x))
x = x/255.0

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, shuffle = False)


seq = Sequential()

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),input_shape=(n_seq, 28, 28, 3),padding='same', return_sequences=True))#n_seq=Noneだった
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),padding='same', return_sequences=False))
seq.add(BatchNormalization())

seq.add(Flatten())#４次元から1次元に変換
seq.add(Dense(1024))#出力が1024
seq.add(Activation('relu'))
seq.add(Dropout(0.5))

seq.add(Dense(1, activation='linear')) # ← ここが出力層！

results = {}

seq.compile(loss='mean_squared_error', optimizer="Adadelta", metrics=['accuracy'])

results[0]= seq.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=16)

model_json_str = seq.to_json()
open('model.json', 'w').write(model_json_str)
#seq.save('ConvLSTM_kaiki_batch16_epoch20.h5');
seq.summary()