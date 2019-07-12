# ShortTermOpenCV

## 执行项目
```bash
git clone https://github.com/Phenanth/ShortTermOpenCV.git
```
克隆本项目后使用`PyCharm`或者`Spider`，或者`Terminal`运行程序

## 项目结构说明
### 文件结构
- `/data_train`：训练数据，五个用户共250张图片
- `/data_train_result`：训练结果模型与人脸特征值数据文件
- `/lib`：调用库
- `/rp`：程序框图
- `/screen_shots`：试运行结果截图，包括人脸特征值与实时识别

### 源代码文件
- `face_acquisition.py`：采集人脸数据
- `face_train.py`：模型训练
- `face_feature.py`：提取人脸特征值
- `face_test.py`：实时识别