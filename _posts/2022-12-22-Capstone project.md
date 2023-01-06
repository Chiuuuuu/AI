---
layout: post
title: PyBullet-Gym for Drones 
author: [johnny, su, chiu]
category: [Lecture]
tags: [jekyll, ai]
---

---
##  PyBullet-Gym for Drones (四軸無人機之強化學習)

**執行環境:**
以下是使用版本供參考
* python 3.9.10
* gym-pybullet-drones-v0.5.2 (儲存.zip檔並解壓縮)
* PyCharm (選擇安裝程序中的所有選項並重新啟動)
* Visual Studio (選擇“使用 C++ 進行桌面開發”)

### PyCharm介紹

PyCharm: [PyCharm ](https://www.jetbrains.com/pycharm/)

**PyCharm** 是一個用於電腦編程的整合式開發環境，主要用於Python語言開發，由捷克公司JetBrains開發，提供代碼分析、圖形化除錯器，整合測試器、整合版本控制系統，並支援使用Django進行網頁開發。<br>
PyCharm是一個跨平台開發環境，擁有Microsoft Windows、macOS和Linux版本。社群版在Apache授權條款下釋出，另外還有專業版在專用授權條款下釋出，其擁有許多額外功能，比如Web開發、Python We框架、Python剖析器、遠端開發、支援資料庫與SQL等更多進階功能。

如果使用 Python 語言進行開發，PyCharm 支援下列幾種辨識功能
1. 項目和代碼導航：專門的項目視圖，文件結構視圖和文件、類、方法和用法之間的快速跳轉。<br>
2. Python 重構：包括重命名、提取方法、引入變量、引入常量、上拉、下壓等。<br>
3. 支持Web框架：Django，web2py和Flask。<br>
4. 集成的Python調試器。<br>
5. Google App Engine Python開發。<br>
6. 版本控制集成：Mercurial，Git，Subversion，Perforce和CVS的統一用戶界面，包含更改列表和合併。<br>
7. 它主要與許多其他面向Python的IDE競爭，包括Eclipse的PyDev和更廣泛的Komodo IDE。<br>

參考來源: [Python自習手札](https://ithelp.ithome.com.tw/articles/10196461)

### 系統簡介及功能說明

1. **系統簡介**：


2. **功能說明**：

---
### 系統方塊圖
系統流程圖<br>
![]()

AI模型說明<br>
![](https://github.com/Chiuuuuu/AI/blob/gh-pages/images/stock_dqn.png?raw=true)

---
### 製作步驟

1.建立資料集dataset<br>
2.移植程式 to kaggle<br>
3.kaggle上訓練模型<br>
4.kaggle上測試模型<br>

---
### 系統測試及成果展示
<iframe width="664" height="498" src="https://www.youtube.com/embed/OP5HcXJg2Aw?list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J" title="【機器學習2021】卷積神經網路 (Convolutional Neural Networks, CNN)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<br>
<br>


**專題實作步驟:**
1. 建立身體動作之姿態照片資料集 (例如：5 poses , take 20 pictures of each pose)<br>
2. 始用**MMPose** 辨識出照片中的各姿勢之身體關鍵點 (use MMPose convert 16 keypoints (x,y) of each pose)<br>
3. 產生姿態關鍵點資料集 x_train.append(pose_keypoints) ( x_train.shape = (20x5, 16, 2), y_train.shape= (20x5, 1) )<br>
4. 建立DNN模型並訓練模型, 然後下載模型檔`pose_dnn.h5`至PC <br>
5. 於PC建立帶camera輸入之服務器程式, 載入模型`pose_dnn.h5`進行姿態動作辨識 <br>

**模型建構與訓練之程式樣本** (PC or Kaggle)<br>

```
input_shape=(16,2)
num_classes=5

inputs = layers.Input(shape=input_shape)
x = layers.Dense(128)(inputs)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs=inputs, outputs=outputs)

models.compile(loss = 'categorical_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])

history = model.fit(x_train, y_train, batch_size=1, epochs=20, validation_data=(x_test, y_test))
models.save_model(model, 'pose_dnn.h5')
```

**姿態辨識服務器之程式樣本** (PC with Camera)<br>

```
model = models.load_model('models/pose_dnn.h5')
labels = ['stand', 'raise-right-arm', 'raise-left-arm', 'cross arms','both-arms-left']

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    mmdet_results = inference_detector(det_model, image) # 人物偵測產生BBox
    person_results = process_mmdet_results(mmdet_results, args.det_cat_id) # 記住人物之BBox  
    pose_results, returned_outputs = inference_top_down_pose_model(...) # 感測姿態產生pose keypoints
    
    x_test = np.array(preson_results).reshape(1,16,2) # 將Keypoints List 轉成 numpy Array
    preds = model.fit(x_test) # 辨識姿態動作
    maxindex = int(np.argmax(preds))
    txt = labels[maxindex]
    print(txt)
```



---
### Hand Pose
![](https://github.com/facebookresearch/InterHand2.6M/blob/main/assets/teaser.gif?raw=true)
**Dataset:** [InterHand2.6M](https://github.com/facebookresearch/InterHand2.6M)<br>

1. Download pre-trained InterNet from [here](https://drive.google.com/drive/folders/1BET1f5p2-1OBOz6aNLuPBAVs_9NLz5Jo?usp=sharing)
2. Put the model at `demo` folder
3. Go to `demo` folder and edit `bbox` in [here](https://github.com/facebookresearch/InterHand2.6M/blob/5de679e614151ccfd140f0f20cc08a5f94d4b147/demo/demo.py#L74)
4. run `python demo.py --gpu 0 --test_epoch 20`
5. You can see `result_2D.jpg` and 3D viewer.

**Camera positios visualization demo**
1. `cd tool/camera_visualize`
2. Run `python camera_visualize.py`


<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

