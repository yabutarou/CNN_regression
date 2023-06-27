# -*- coding: utf-8 -*-
import os
import cv2

def main():
    data_dir_path = u"./cifar100/1/"
    file_list = os.listdir(r'./cifar100/1/')

    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.png' or u'.jpeg' or u'.jpg':
            abs_name = data_dir_path + '/' + file_name
            image = cv2.imread(abs_name)
            #以下各画像に対する処理を記載する
            figure = image[3:33,6:28]#(y,x)
            cv2.imwrite(str(file_name),figure)

if __name__ == '__main__':
    main()