## 实训作业二：TensorFlow Schoolwork说明文档

### 算法介绍：

K-近邻算法（K Nearest Neighbor），是最基本的分类算法，属于懒惰学习的一种，它没有显式的学习过程。其基本思想是采用测量不同样本之间的距离来进行分类。 

### 算法原理：

首先，存在一个已知类别标签样本数据集合（训练集）。在输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较（计算距离），然后提取样本集中特征最相似的前K个数据（K近邻）的类别标签。选择这K个样本中类别次数出现最多的类别标签作为此输入数据的预测标签。 

### 算法实现：

#### 加载数据集：

```python
# 导入必要的模型包
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 加载鸢尾花数据集
from sklearn.datasets import load_iris
iris = load_iris()
data = iris.data
label = iris.target
feature_names = iris.feature_names
```

#### 划分数据集

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(                       												 data,label,test_size=0.2,shuffle=True,random_state=6)
```

#### 模型构建及学习
```python
# 构建模型，也就是计算输入测试样本和所有训练样本之间的距离
# x为训练集的占位符，ｘ_为一个验证样本的占位符
x = tf.placeholder(tf.float32, shape=[None, 4])
x_ = tf.placeholder(tf.float32, shape=[4])
dist = tf.sqrt(tf.reduce_sum(tf.abs(tf.add(x, tf.negative(x_))), 1))
# 训练模型
def train_knn(K): #超参数K
    with tf.Session() as sess:
        pred = [] #存放所有测试样本的预测类别
        for i in range(len(X_test)):
            dist_mat = sess.run(dist, feed_dict={x:X_train, x_:X_test[i]})  
            # 将距离矩阵排序后，取出前Ｋ个近邻
            knn_idx = np.argsort(dist_mat)[:K]       
            # 按这Ｋ个近邻的类别标记进行投票，得出ｘ_的预测标记值y_pred
            classes = [0, 0, 0]
            for idx in knn_idx:
                if(y_train[idx]==0):
                    classes[0] += 1
                elif(y_train[idx]==1):
                    classes[1] += 1
                else:
                    classes[2] += 1
            y_pred = np.argmax(classes)
            pred.append(y_pred)
        return pred
```
#### 模型验证

```python
# 模型评估：根据正确率，选出最优超参数K      
def valid_knn():
    k_scores = []
    k_range = range(1, 31)
    for K in k_range:
        y_pred = train_knn(K)
        y_true = y_test
        acc = np.sum(np.equal(y_pred,y_true)) / len(y_true)
        k_scores.append(acc)
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Accuracy')   
# 由下方图像可知最优K值以及在验证集上的最高正确率
valid_knn()
```

![acc](.\images\acc.png)

------

姓名：赵永权
学号：2016011715
学院：软件学院
专业：软件工程

