import numpy as np
import pandas as pd
 
# 读取鸢尾花数据集，header参数来指定标题的行。默认为0。如果没有标题，则使用None。
data = pd.read_csv("iris.csv",header=0)
# 显示前n行记录。默认n的值为5。
#data.head()
# 显示末尾的n行记录。默认n的值为5。
#data.tail()
# 随机抽取样本。默认抽取一条，我们可以通过参数进行指定抽取样本的数量。
# data.sample(10)
# 将类别文本映射成为数值类型
 
data["species"] = data["species"].map({"setosa": 0, "versicolor": 1, "virginica": 2})
# 删除不需要的Id列。
# data.drop("Id", axis=1, inplace=True )
# data.drop_duplicates(inplace=True)
## 查看各个类别的鸢尾花具有多少条记录。
data["species"].value_counts()

#构建训练集与测试集，用于对模型进行训练与测试。
# 提取出每个类比的鸢尾花数据
t0 = data[data["species"] == 0]
t1 = data[data["species"] == 1]
t2 = data[data["species"] == 2]
# 对每个类别数据进行洗牌 random_state 每次以相同的方式洗牌 保证训练集与测试集数据取样方式相同
t0 = t0.sample(len(t0), random_state=0)
t1 = t1.sample(len(t1), random_state=0)
t2 = t2.sample(len(t2), random_state=0)
# 构建训练集与测试集。
train_X = pd.concat([t0.iloc[:40, :-1], t1.iloc[:40, :-1], t2.iloc[:40, :-1]] , axis=0)#截取前40行，除最后列外的列，因为最后一列是y
train_y = pd.concat([t0.iloc[:40, -1], t1.iloc[:40, -1], t2.iloc[:40, -1]], axis=0)
test_X = pd.concat([t0.iloc[40:, :-1], t1.iloc[40:, :-1], t2.iloc[40:, :-1]], axis=0)
test_y = pd.concat([t0.iloc[40:, -1], t1.iloc[40:, -1], t2.iloc[40:, -1]], axis=0)

#定义KNN类，用于分类，类中定义两个预测方法，分为考虑权重不考虑权重两种情况
class KNN:
 ''' 使用Python语言实现K近邻算法。（实现分类） '''
 def __init__(self, k):
  '''初始化方法 
   Parameters
   -----
   k:int 邻居的个数
  '''
  self.k = k
 
 def fit(self,X,y):
  '''训练方法
   Parameters
   ----
   X : 类数组类型，形状为：[样本数量, 特征数量]
   待训练的样本特征（属性）
  
  y : 类数组类型，形状为： [样本数量]
   每个样本的目标值（标签）。
  '''
  #将X转换成ndarray数组
  self.X = np.asarray(X)
  self.y = np.asarray(y)
  
 def predict(self,X):
  """根据参数传递的样本，对样本数据进行预测。
  
  Parameters
  -----
  X : 类数组类型，形状为：[样本数量, 特征数量]
   待训练的样本特征（属性） 
  
  Returns
  -----
  result : 数组类型
   预测的结果。
  """
  X = np.asarray(X)
  result = []
  # 对ndarray数组进行遍历，每次取数组中的一行。
  for x in X:
   # 对于测试集中的每一个样本，依次与训练集中的所有样本求距离。
   dis = np.sqrt(np.sum((x - self.X) ** 2, axis=1))
   ## 返回数组排序后，每个元素在原数组（排序之前的数组）中的索引。
   index = dis.argsort()
   # 进行截断，只取前k个元素。【取距离最近的k个元素的索引】
   index = index[:self.k]
   # 返回数组中每个元素出现的次数。元素必须是非负的整数。【使用weights考虑权重，权重为距离的倒数。】
   count = np.bincount(self.y[index], weights= 1 / dis[index])
   # 返回ndarray数组中，值最大的元素对应的索引。该索引就是我们判定的类别。
   # 最大元素索引，就是出现次数最多的元素。
   result.append(count.argmax())
  return np.asarray(result)

  #创建KNN对象，进行训练与测试。
knn = KNN(k=5)
#进行训练
knn.fit(train_X,train_y)
#进行测试
result = knn.predict(test_X)
# display(result)
# display(test_y)
print(np.sum(result == test_y))
print(np.sum(result == test_y)/ len(result))

#导入可视化所必须的库。
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False
 
#绘制散点图。为了能够更方便的进行可视化，这里只选择了两个维度（分别是花萼长度与花瓣长度）。
# {"Iris-virginica": 0, "Iris-setosa": 1, "Iris-versicolor": 2})
# 设置画布的大小
plt.figure(figsize=(10, 10))
# 绘制训练集数据
plt.scatter(x=t0["sepal_length"][:40], y=t0["petal_length"][:40], color="r", label="Iris-virginica")
plt.scatter(x=t1["sepal_length"][:40], y=t1["petal_length"][:40], color="g", label="Iris-setosa")
plt.scatter(x=t2["sepal_length"][:40], y=t2["petal_length"][:40], color="b", label="Iris-versicolor")
# 绘制测试集数据
right = test_X[result == test_y]
wrong = test_X[result != test_y]
plt.scatter(x=right["sepal_length"], y=right["petal_length"], color="c", marker="x", label="right")
plt.scatter(x=wrong["sepal_length"], y=wrong["petal_length"], color="m", marker=">", label="wrong")
plt.xlabel("sepal_length")
plt.ylabel("petal_length")
plt.title("KNN")
plt.legend(loc="best")
# plt.savefig("./KNN.png")
plt.show()