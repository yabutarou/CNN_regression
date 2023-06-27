import keras
import sys, os
import numpy as np
from PIL import Image
from keras.models import load_model

imsize = (28, 28)#横,縦


testpic     = "9(4373).jpg"
keras_param = "weights_kaiki_test_batch16_epoch20_nitika.h5"

def load_image(path):
    img = Image.open(path)
    img = img.convert('RGB')
    # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
    img = img.resize(imsize)
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img/255.0
    return img

model = load_model(keras_param)
img = load_image(testpic)
prd = model.predict(np.array([img]))
#prd = model.predict(img)
print("答え",prd) # 精度の表示
