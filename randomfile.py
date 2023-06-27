#ランダムに抽出し別ファイルを作成
import glob
import random
import os 
import shutil
import math

number = "9"
INPUT_DIR = 'cifar230120_re_resolution_prydown2' + '/' + number
OUTPUT_DIR = 'cifar230120_high_and_low'

#ランダムで抽出する割合
SAMPLING_RATIO = 0.5

def random_sample_file():
    files = glob.glob(INPUT_DIR + '/*.jpg')

    random_sample_file = random.sample(files,math.ceil(len(files)*SAMPLING_RATIO))#fileからランダムに#math.ceil(数字の切り上げ)
    os.makedirs(OUTPUT_DIR + "/" + number ,exist_ok=True)

    for file in random_sample_file:
        shutil.copy2(file,OUTPUT_DIR + "/" + number + "/")#元のファイル と同じコンテンツ (データ) をもつファイルを dst としてコピーし作成

if __name__ == '__main__':
    random_sample_file()