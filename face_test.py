# -*- coding: utf-8 -*-
import cv2
import time
from collections import Counter
subjects = ['unKnown', 'mx','crr','wj','gzd','cwf']
# 创建人脸识别器
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
 #------或需更改------
cascPath = "./lib/haarcascade_frontalface_default.xml" 
faceCascade = cv2.CascadeClassifier(cascPath)

# 读取训练集
#------或需更改------
face_recognizer.read('./data_train_result/data_train_result.xml') 
# 打开视频捕获设备
#t为视频捕获设备添加try模块，防止异常 
try:
    cap = cv2.VideoCapture(0)

    #label_text识别出的label对应人名
    #count记录采集到人脸的帧数，作为每个十帧结点
    #frame_array保存十帧里识别人脸对应的label   
    label_text = ''
    count = 1
    frame_array = []
    
    while True:
        if not cap.isOpened():
            print('Unable to load camera.')
            time.sleep(5) 
            pass 
        # 读视频帧 
        ret, frame = cap.read()
        # 转为灰度图像 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         # 调用分类器进行检测 
        faces = faceCascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(200, 200) 
        )
            
        for (x, y, w, h) in faces:
            #预测
            #conf 为可能性
            label,conf = face_recognizer.predict(gray[y:y+h, x:x+w])
            
            #是否能识别出人脸的可能性随着训练数据集会发生变化
            #------或需更改------
            #可能性小于80，视为未识别出，反之则识别出
            if conf < 80 :
                frame_array.append(0)
            else:
                frame_array.append(label)

            #print(label,conf)
            #隔十帧识别
            if count%10 == 0 :
                label_text = subjects[Counter(frame_array).most_common(1)[0][0]]   #选取frame_array出现次数最多的元素
                #print(frame_array)
                #print(label_text)

                frame_array = []  
            #绘制，第一个十帧不显示
            if  count/10 > 0:
                cv2.rectangle(frame, (x, y, w, h), (128, 0, 0), 4)
                cv2.putText(frame, label_text, (x+w//2,y-20),cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 0), 4)    
            count += 1
        #显示视频
        cv2.imshow('Video', frame)
        #退出
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
#BaseException为所有异常基类
except BaseException as err:
    print(err)   
# 关闭摄像头设备 
cap.release() 
# 关闭所有
cv2.destroyAllWindows()


                
            
        