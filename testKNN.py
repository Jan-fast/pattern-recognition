import operator
import random

import numpy as np
import sklearn.datasets as sd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 读取数据集
iris = sd.load_iris()  # 载入数据据
# print(iris['target_names'])  # 'target_names': array(['setosa', 'versicolor', 'virginica']
# print(iris[
#           'feature_names'])  # 'feature_names'：['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x_data = iris['data']
y_data = iris['target']
features = iris['feature_names']
labels = iris['target_names']

# 自动切分数据集方法
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

def split_dataset(x_data, y_data, test_size=40):
    # 手动打乱数据集

    data_size = len(x_data)  # 获取行数
    id_ = [i for i in range(data_size)]  # 列表解析式建立index列表
    random.shuffle(id_)  # random.shuffle打乱顺序
    x_data = x_data[id_]  # 数据集调整
    y_data = y_data[id_]

    # 手动切分数据集方法

    x_train = x_data[test_size:]  # 40：150为训练集
    y_train = y_data[test_size:]
    x_test = x_data[:test_size]  # 0：40为测试集
    y_test = y_data[:test_size]
    return x_train, y_train, x_test, y_test



def KNN(x_test_, x_train, y_train, K):
    # 定义KNN函数

    x_test_0 = np.tile(x_test_, (len(x_train), 1))  # 复制x_test[0],用于计算delta
    delta_mat = x_train - x_test_0  # 计算差值
    delta2 = delta_mat ** 2  # 计算平方
    distance2 = []  # 求和
    for i in range(len(delta2)):
        distance2.append(sum(delta2[i]))

    distance = np.sqrt(distance2)  # 开方
    sorted_distance = distance.argsort()#对distance进行排序
    # print(sorted_distance)
    dict_ = {}
    for i in range(K):#取前K个数据
        label_ = y_train[sorted_distance[i]]
        dict_[label_] = dict_.get(label_, 0) + 1  # 没有则设为0；有则+1
    # sorted_dict = sorted(dict_, key=dict_.__getitem__, reverse=True)  # 对字典进行排序
    # 或者
    sorted_dict = sorted(dict_.items(), key=operator.itemgetter(1), reverse=True)  # 返回list[(1, 3), (0, 2)]
    return sorted_dict[0][0]

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = split_dataset(x_data, y_data, test_size=40)
    prediction = []
    for i in range(len(x_test)):
        x_test_ = x_test[i]
        label = KNN(x_test_, x_train, y_train, K=20)
        prediction.append(label)
    print(prediction)
    print(y_test)
    print(classification_report(y_test, prediction))  # 精度报告
    print(confusion_matrix(y_test, prediction))

