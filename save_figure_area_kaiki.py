
import cv2
import function_Program as function
import time
import math
import os
import numpy as np
from PIL import Image

a=0
judge = 0
idx = 0
Picture_count = 0
pfd_de = 0
charlist11 = []
charlist12 = []
stratTime = time.time() 


#以下メインプログラム
if __name__ == "__main__":#インポートされた際にプログラムが動かないようにするために、以下のように if __name__ == "__main__": というif文を書く。　単体のプログラムの時は関係ない
    #情報入力
    print("PLEASE ENTER VIDEO NAME (NO EXTENSION) : ")
    
    video_name = input()
    
    path = video_name + '.txt'
    
    dir_name = video_name + "_dst"+ "_figure_area" + "_cockpit"+  "_dir"
    
    if not os.path.isdir(dir_name):   #保存先ファイルがないなら作成
    	os.makedirs(dir_name)
        
    path = dir_name + "/" + path
    
    cap = cv2.VideoCapture(video_name + ".mp4")
    
        #動画終了までループ
    while(cap.isOpened()):
        idx += 1    #この部分でフレーム間隔を調整 2にすると約2倍時間かかる　1の時１秒一回
        ret,frame = cap.read()
        #frame = cv2.rotate(frame, cv2.ROTATE_180)
        if ret == True:
            frame3 = frame.copy() #類似度判定用のフレームコピー
            
            #cv2.imwrite(dir_name + "/" + str(Picture_count) + ".jpg", frame)
            #Picture_count += 1

            pfd_de+= 1 #試行回数　分母
            # input image
            frame = cv2.resize(frame, (1920, 1080)) 
            #======================↓↓↓二値化↓↓↓======================
            imgray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#グレースケールへの変換
            thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,4) #適応的二値化処理

            #ゴマ塩ノイズ除去
            gsize = 3#ノイズ除去のパラメータ
            thresh = cv2.medianBlur(thresh,gsize)#ノイズ除去
            #======================↑↑↑二値化↑↑↑======================
            #類似度検出用配列の初期値
            distlen_0,distlen_1, distret_0, distret_1 = [],[],[],[] 
            #======================↓↓↓矩形の角座標↓↓↓======================
            #輪郭の推定
            contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            img = frame


            #射影変換
            #areas = np.array([[713,1203],[770,1555],[1277,1533],[1233,1179]])#高画質221017
            #areas = np.array([[351,600],[382,777],[635,768],[614,587]])#低画質221017
            #areas = np.array([[511,1224],[572,1608],[1116,1594],[1076,1213]])#高画質
            #areas = np.array([[256,612],[286,804],[558,797],[538,607]])#低画質
            #areas = np.array([[316,246],[439,1000],[1478,974],[1395,223]])#低画質(範囲変更後)
            areas = np.array([[1332,1096],[1428,1895],[2462,1744],[2402,975]])#高画質221109
            pts1 = np.float32(areas)  #出力前の4点の座標
            pts2 = np.float32([[0,0],[0,850],[1160,850],[1160,0]])    #出力後の4点の座標
            M = cv2.getPerspectiveTransform(pts1,pts2)  #3*3の変換行列
            dst = cv2.warpPerspective(frame3,M,(1160,850))
            h, w, c = dst.shape
            #IAS = dst[int(h*(250/768)):int(h*(330/768)), int(w*(160/1024)):int(w*(250/1024))]#160,250
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:
                IAS = dst[int(h*(235/768)):int(h*(325/768)), int(w*(130/1024)):int(w*(245/1024))]#221109
                cv2.imwrite(dir_name + "/" + str(Picture_count) + ".jpg", IAS)
                Picture_count += 1
            elif idx < cap.get(cv2.CAP_PROP_FPS): #1秒間隔の間　動画確認時のみ使用
                continue
            else:
                IAS = dst[int(h*(235/768)):int(h*(325/768)), int(w*(130/1024)):int(w*(245/1024))]#221109
                cv2.imwrite(dir_name + "/" + str(Picture_count) + ".jpg", IAS)
                Picture_count += 1
                idx = 0
        break