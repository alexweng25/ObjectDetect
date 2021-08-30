# Object Detection

採用BoW(SURFT+Clusting)+SVM(Support Vector Machine)方式進行物件辨識。
先行對影像進行SURFT特徵擷取，接著使用BOVW進行資料降維建立物件特徵模型(Bag Of Visual Word)，BOVW主要採用K-Means進行特徵分群。
最後使用SVM進行分類器的訓練，分類器採一對多進行訓練(SVC裡面有一個參數為decision_function_shape : ‘ovo’, ‘ovr’, default=’ovr’, ovr 就是採一對多方式)

- [ ] train_svm.py為訓練模型檔案，利用GridSearchCV找尋SVM的最佳參數
- [ ] test_svm.py為辨識影像檔案
- [ ] ALL_grid_SVM.pkl為SVM分類器的模型存檔
- [ ] BOWCluster.pkl為BOW分群演算法的模型存檔
- [ ] image-augmentation-with-opencv.py為影像增強的檔案。目的是為了解決物件旋轉問題。

在TEST的資料夾內我放置的都是各類別的全部檔案，分別進行測試。可以改test_svm.py把全部檔案一起進行測試。
