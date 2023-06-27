import keras
import sys, os
import numpy as np
from PIL import Image
from keras.models import load_model
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
model = load_model("ConvLSTM_kaiki_batch16_epoch20.h5")
test_dir ="221109_kirinuki_20230329_dir"
folder = os.listdir(test_dir)
#元画像640x480をresizeする際の大きさ設定
x_image_size = 28#640→200#横
y_image_size = 28#480→200#縦
dense_size  = len(folder)
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
n_seq = 5
n_sample = imgs.shape[0] - n_seq
print(n_sample)

x = np.zeros((n_sample, n_seq, y_image_size, x_image_size, 3))
for i in range(n_sample):
    x[i] = imgs[i:i+n_seq]
x_train = x/255.0
predictions = model.predict(x_train)
predictions = predictions*255.0
print(len(predictions))
print(predictions)