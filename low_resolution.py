#フォルダ内一括編集（低画質化）
# -*- coding: utf-8 -*-
import os
import cv2
import glob

def main():
    input_path = 'cifar230120 (1)'#保存前フォルダ名
    output_path = 'cifar230120_re_resolution_prydown1'#保存先フォルダ名
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
                dst = cv2.pyrDown(image)
                #dst = cv2.pyrDown(dst)
                dst = cv2.resize(dst, (image.shape[1], image.shape[0]),interpolation=cv2.INTER_LINEAR)
                
                

                if not os.path.exists(dirname):
                    os.mkdir(dirname)
                cv2.imwrite(os.path.join(dirname, str(file_name)),dst)
    
        
if __name__ == '__main__':
    main()