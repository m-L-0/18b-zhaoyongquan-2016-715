## 实训作业三：Classification Schoolwork说明文档

### 光谱数据分类：

#### 加载训练集及其类别标记
``` python
#导入需要的python模块
import scipy.io as sio
import numpy as np
import os
import re
# 加载数据集及其类别标签
path = '../dataset/train/'
data = np.array([])
label = []
for file in os.listdir(path):
    key = re.sub('.mat',"",file)
    y = re.sub("\D","",file)
    a = sio.loadmat(path+file)[key]
    class_num = a.shape[0]
    for _ in range(class_num):
        label.append(y)
    data = np.append(data,a)  
data = data.reshape((-1,200))    
label = np.array(label,dtype=np.int32)
```

#### 划分训练集、验证集
``` python
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(data,label,test_size=0.3,shuffle=True)
```

#### 数据规范化预处理
``` python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #归一化
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
```

#### 格点搜索寻找最佳参数组合
``` python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
svc = SVC(class_weight='balanced')
grid = GridSearchCV(svc,
                    param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=5)
grid.fit(X_train, y_train)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
```

#### 构建分类器并训练模型
``` python
from sklearn.metrics import accuracy_score
svc = SVC(C=10,gamma=0.01,class_weight='balanced')  #构建分类器，设定参数
svc.fit(X_train,y_train)
y_pred = svc.predict(X_valid)
print(accuracy_score(y_valid,y_pred))
# 0.9475457170356112
```


#### 加载测试集并做相应处理
``` python
# 加载测试集
test_data = sio.loadmat("../dataset/data_test_final.mat")['data_test_final']
test_data = np.array(test_data, dtype=np.float64)
x_test = scaler.transform(test_data)
```


#### 生成测试集标签并将其导出为csv文件
``` python
import pandas as pd
y_test = svc.predict(x_test)
data = pd.DataFrame(y_test)
data.to_csv("../dataset/test_labels.csv")
```

------

姓名：赵永权

学号：2016011715

学院：软件学院

专业：软件工程