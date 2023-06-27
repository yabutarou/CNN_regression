import keras
import sys, os
import numpy as np
from PIL import Image
from keras.models import load_model

imsize = (28, 28)


testpic     = "6_test.png"
keras_param = "weights230120_test_prydown1_batch16_epoch20.h5"

def load_image(path):
    img = Image.open(path)
    img = img.convert('RGB')
    # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
    img = img.resize(imsize)
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0
    return img

model = load_model(keras_param)
img = load_image(testpic)
prd = model.predict(np.array([img]))
print(prd) # 精度の表示
prelabel = np.argmax(prd, axis=1)
if prelabel == 0:
    print("0")
elif prelabel == 1:
    print("1")
elif prelabel == 2:
    print("2")
elif prelabel == 3:
    print("3")
elif prelabel == 4:
    print("4")
elif prelabel == 5:
    print("5")
elif prelabel == 6:
    print("6")
elif prelabel == 7:
    print("7")
elif prelabel == 8:
    print("8")
elif prelabel == 9:
    print("9")
