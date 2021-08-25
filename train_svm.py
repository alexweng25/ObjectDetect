# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 20:01:09 2019

@author: Alex
"""

import os
import numpy as np
import cv2
#from imutils import paths
from sklearn.externals import joblib
from sklearn import cluster
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import pickle
from pytictoc import TicToc

class objectSVM(object):
    def __init__(self, trainpath, featurs = 400, num_clusters = 1000):
        self.trainpath = trainpath
        self.num_clusters = num_clusters
        self.ClassesLabel = ''
        self.featurs = featurs
        
    def path(self, rela, Label, Index):
        if rela == 'pos':
            return '%s/%s/%d.jpg' % (self.trainpath, Label, Index + 1)
        else:
            return '%s/%s/%s/%d.jpg' % (self.trainpath, Label, 'other', Index + 1)
    
    def fitBOW(self, train_path, k):
        # create Bag of words using k-means clustering
        self.bow_img_descriptor_extractor = cluster.MiniBatchKMeans(n_clusters=self.num_clusters, random_state=0)

        feature_des = []
        arr = np.arange(0,300,3)
        #arrData = np.random.choice(arr, 100, replace=False)
        ClassesLabels = os.listdir(trainpath)
        
        #Get images from each target for k-means clustering
        for label in ClassesLabels:
            for i in arr:
                feature = self.feature_detect(self.path('pos', label, i))
                feature_des.append(feature)

        feature_des = np.asarray(feature_des)
        feature_des = np.concatenate(feature_des, axis=0)
        self.bow_img_descriptor_extractor.fit(feature_des)
        
        # Save BOW model
        pickle.dump(self.bow_img_descriptor_extractor, open("BOWCluster.pkl", "wb"))
        
        
    def fitSVM(self, Label):  
        pos = 'pos'
        neg = 'other'
        if Label == 'ALL_grid':
            fileList = []
            for root, subFolders, files in os.walk(self.trainpath):
                fileList.append(len(files))
            fileList.remove(0)    
            
            traindata, trainlabels = [], []
            i = 0
            ClassesLabels = os.listdir(self.trainpath)
            for label in ClassesLabels:
                arr = np.arange(fileList[i])
                arrData = np.random.choice(arr, 200, replace=False)
                i = i + 1
                for number in arrData:
                    traindata.append(self.bow_descriptor_extractor(self.path(pos, label, number)))
                    trainlabels.append(i)
        else:
            # 創建兩個數組，分別對應訓練數據和標籤，並用BOWImgDescriptorExtractor產生的描述符填充
            # 按照下面的方法生成相應的正負樣本圖片的標籤 1：正匹配  -1：負匹配
            traindata, trainlabels = [], []
            for i in range(10):  # 這裏取N張圖像做訓練
                traindata.append(self.bow_descriptor_extractor(self.path(pos, Label, i)))
                trainlabels.append(1)
                traindata.append(self.bow_descriptor_extractor(self.path(neg, Label, i)))
                trainlabels.append(-1)
            
        trainlabels= np.array(trainlabels).reshape((-1, 1))
        
        # 分7,3
        x_train, x_test, y_train, y_test = train_test_split(traindata, trainlabels, test_size=0.3, random_state=0)
        
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['auto', 0.001, 0.005, 0.0001, 0.0005],
                     'C': [1, 10, 50, 100, 500], 'class_weight' : ['balanced']}]
        scores = ['precision', 'recall']
        
        clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % scores[0])
        clf.fit(x_train, y_train)
               
        model = clf.best_estimator_
        model.fit(x_train, y_train)
        
        y_testpredict = clf.predict(x_test)
        #print(classification_report(y_true, y_pred))
        
        # 創建一個SVM對象
        #self.svmmodel = SVC()
        # 使用訓練數據和標籤進行訓練
        #self.svmmodel.fit(x_train, y_train)
        
        #r_squared = self.svmmodel.score(x_train, y_train)
        #print(f'r_squared: {r_squared}')
        #y_testpredict = self.svmmodel.predict(x_test)
        accuracy = accuracy_score(y_test, y_testpredict)
        print(f'accuracy: {accuracy}')
        #joblib.dump(self.svmmodel, "{}_SVM.pkl".format(Label))

        #  列印最佳參數
        print("Best estimator:")
        print(clf.best_estimator_)
        print()
        print("Best score::")
        print(clf.best_score_)
        print()
        
        #  設置陣列的資料
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        #  這邊依設置的參數列印出所有參數的得分狀況
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        print(classification_report(y_test, y_testpredict))
        print()
        
        joblib.dump(clf, "{}_SVM.pkl".format(Label))

    def feature_detect(self, imgpath):
        img = cv2.imread(imgpath)
        #print(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # find the key-points with ORB
        #kp = self.orb.detect(img, None)
        #print(imgpath, len(kp))
        # compute the descriptors with ORB
        #kp, des = self.orb.compute(img, kp)
        
        # find the key-points with SURF
        surf = cv2.xfeatures2d.SURF_create(self.featurs)
        kp, des = surf.detectAndCompute(img,None)
        #print(len(kp))
        return des
    
    def bow_descriptor_extractor(self, img_path):
        # compute the descriptors with ORB
        des = self.feature_detect(img_path)
        # classification of all descriptors in the bow k-means model
        self.bow_img_descriptor_extractor = pickle.load(open("BOWCluster.pkl", 'rb'))
        predict = self.bow_img_descriptor_extractor.predict(des)
        # calculates the histogram
        hist, _ = np.histogram(predict, bins=self.num_clusters)

        return hist

if __name__ == '__main__':
    trainpath = './traindata'
    t = TicToc() #create instance of class
    ClassesLabels = os.listdir(trainpath)
    #print(ClassesLabels)
    
    # objectSVM(training image path, SURF features)
    #測試影像路徑，SURF影像特徵數量，BOW群集數量
    a = objectSVM(trainpath, 400, 2000)    
    
    # fitBOW( path, kmeans 群數)
    #t.tic() #Start timer
    #a.fitBOW(trainpath, 2000)
    #t.toc() #Time elapsed since t.tic()
    #BOW_elapsed = t.tocvalue()
    
    #for ClassesLabel in ClassesLabels:
    t.tic() #Start timer
    a.fitSVM('ALL_grid')
    t.toc() #Time elapsed since t.tic()
    BOW_elapsed = t.tocvalue()
        
