import cv2
import dlib
import numpy as np

'''
https://www.cnblogs.com/AdaminXie/p/8137580.html
'''

datafilename = './data_train_result/features.txt'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./lib/shape_predictor_68_face_landmarks.dat')

names = ['./data_train/s1/User.01.1.jpg',
         './data_train/s2/User.02.1.jpg',
         './data_train/s3/User.03.01.jpg',
         './data_train/s4/User.4.1.jpg',
         './data_train/s5/User.5.1.jpg']
imgs = []
for filename in names:
    img = cv2.imread(filename)
    imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

font = cv2.FONT_HERSHEY_SIMPLEX

f = open(datafilename, 'w', encoding='utf-8')
count = 0
for gray in imgs:
    count += 1
    faces = detector(gray, 0)
    if len(faces) != 0:
        for i in range(len(faces)):
            # 取坐标
            landmarks = np.matrix([[p.x, p.y] for p in predictor(gray, faces[i]).parts()])
            # 根据坐标进行绘制
            for index, point in enumerate(landmarks):
                # 写入文件
                if index == 0:
                    f.write('User '+ str(count) + '\n')
                f.write('x: ' + str(point[0, 0]) + ', y:' + str(point[0, 1]) +  '\n')

                pos = (point[0, 0], point[0, 1])
                cv2.circle(gray, pos, 2, color=(139, 0, 0))
                cv2.putText(img, str(index + 1), pos, font, 0.2, (183, 244, 244), 1, cv2.LINE_AA)
    else:
        # 无人脸
        cv2.putText(gray, "no face detected.", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

    # 对显示窗口的指定
    cv2.namedWindow('Image', 0)
    # 显示窗口
    cv2.imshow("Image", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

f.close()