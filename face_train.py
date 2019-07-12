# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
def prepare_training_data(data_folder_path):

    # 获取文件夹中的目录
    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []

    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;

        label = int(dir_name.replace('s',''))

        subject_dir_path = data_folder_path + "/" + dir_name

        subject_images_names = os.listdir(subject_dir_path)

        # 检测脸部并将脸部添加到列表
        for image_name in subject_images_names:

            if image_name.startswith("."):
                continue;

            # 建立图像路径
            image_path = subject_dir_path + "/" + image_name

            image = cv2.imread(image_path)
            cv2.imshow(str(label),image)
            cv2.waitKey(100)

            # face, rect = detect_face(image)
            face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 将脸添加到脸部列表
            faces.append(face)
            # 为这张脸添加标签
            labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels

print("Preparing data...")
#------或需更改------
faces, labels = prepare_training_data("./data_train")
print("Data prepared")

# 打印面孔数量及标签
print("Total faces:", len(faces))
print("Total labels:", len(labels))

# 创建人脸识别器
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# 训练面部识别器
face_recognizer.train(faces, np.array(labels))
#------或需更改------
face_recognizer.save('./data_train_result/data_train_result.xml')

