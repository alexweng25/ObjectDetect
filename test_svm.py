# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:18:20 2019

@author: ASUS
"""
from sklearn.externals import joblib
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import os
import cv2
from sklearn import cluster, metrics
import pickle
import pandas as pd # 引用套件並縮寫為 pd  
from skimage import transform
import matplotlib.pyplot as plt

from pytictoc import TicToc
print("Testing the SVM classifier")  

folder_path = './test/'
        
def path(pos, Label, Index):
    if pos == 'pos':
        return '%s/%s/%d.jpg' % (folder_path, Label, Index + 1)
    else:
        return '%s/%s/%s/%d.jpg' % (folder_path, Label, 'other', Index + 1)
    
def surf_detect( imgpath, feature):
    img = cv2.imread(imgpath)
    #print(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #影像變大為了方便取特徵，增加辨識度
    img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_CUBIC)
     # create the key-points producer
    surf = cv2.xfeatures2d.SURF_create(feature)
    # compute the descriptors with SURF producer
    kp, des = surf.detectAndCompute(img,None)
    
    img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),2)
    plt.imshow(img2)
    plt.show()
    #print(len(kp))
    return des
    
def predict(model, img_path, num_clusters, feature = 400):
    # compute the descriptors with ORB
    des = surf_detect(img_path, feature)
    
    # classification of all descriptors in the bow k-means model
    predict = model.predict(des)
    # calculates the histogram
    hist, _ = np.histogram(predict, bins=num_clusters)

    return hist

def sliding_window(img, patch_size, istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch

# 找各類別的影像總數量
fileList = []
trainpath = './traindata'
for root, subFolders, files in os.walk(trainpath):
    fileList.append(len(files))
fileList.remove(0)
ClassesLabels = os.listdir(trainpath)
# Load model
BOW = pickle.load(open("BOWCluster.pkl", 'rb')) 
model = joblib.load("ALL_grid_SVM.pkl")
cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
                    
predict_result = []
test_class = []
t = TicToc() #偵測時間
BOW_elapsed = []
for i in range(21, 29):
    t.tic() #Start timer
    #讀圖並用BOW獲得影像描述子，2000為訓練時給定的群集數量
    
    test_feature = predict(model = BOW, img_path = "./test/{}.jpg".format(i), num_clusters = 2000)
    test_feature= np.array(test_feature).reshape((1, -1))
    #利用SVC預測結果
    a=model.predict(test_feature)
    t.toc() #Time elapsed since t.tic()模型進行預測
    BOW_elapsed.append( t.tocvalue())
    
    predict_result.append(a)
    img = cv2.imread("./test/{}.jpg".format(i))
    if a==3:
        cv2.putText(img, "noodle", (10, 30), cv2. FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2, cv2. LINE_AA)
    elif a==2:
        cv2.putText(img, "lays", (10, 30), cv2. FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2, cv2. LINE_AA)
    else:
        cv2.putText(img, "Drink", (10, 30), cv2. FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2, cv2. LINE_AA)

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    #添加標記
    test_class.append(2)


   
score=accuracy_score(np.asarray(test_class), predict_result)    
print(score)
averagetime = sum(BOW_elapsed) / len(BOW_elapsed)
print(f'average time: {averagetime}')
#predict_result =  [] 
#for model in models:
#    predict_result.append(model.predict(test_feature))

    
#score=accuracy_score(np.asarray(test_class), predict)
