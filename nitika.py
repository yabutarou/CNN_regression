#フォルダ内一括編集（２値化）
# -*- coding: utf-8 -*-
import os
import cv2
import glob
def main():
    input_path = 'cifar230120'#保存前フォルダ名
    output_path = 'cifar230120_2tika'#保存先フォルダ名
    list1 = [os.path.basename(p) for p in glob.glob(str(input_path) + '/*')]
    for i in list1:
        file_list = os.listdir(r'./' + str(input_path) + '/' + str(i))
        data_dir_path = u"./" + str(input_path) + "/" + str(i)
        dirname = str(output_path) + "/" + str(i)
        for file_name in file_list:
            root, ext = os.path.splitext(file_name)
            if ext == u'.png' or u'.jpeg' or u'.jpg':
                abs_name = data_dir_path + '/' + file_name
                print(abs_name)
                image = cv2.imread(abs_name)
                #以下各画像に対する処理を記載する
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#RGB?
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
                
                if not os.path.exists(dirname):
                    os.mkdir(dirname)
                cv2.imwrite(os.path.join(dirname, str(file_name)),thresh)
    
        
if __name__ == '__main__':
    main()