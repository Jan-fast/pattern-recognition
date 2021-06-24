import numpy as np
import treePlotter
import pandas as pd

def binarySplitDataSet(dataset,feature,value):
    '''
    输入：数据集，数据集中某一特征列，该特征列中的某个取值
    功能：将数据集按特征列的某一取值换分为左右两个子数据集
    输出：左右子数据集
    '''
    matLeft = dataset.loc[dataset[feature] <= value]
    matRight = dataset.loc[dataset[feature] > value]
    return matLeft,matRight
    
def regressLeaf(dataset):
    '''
    输入：数据集
    功能：求数据集输出列的均值
    输出：对应数据集里数量最多的种类
    '''
    count0 = len(dataset[dataset['species'] == 0])
    count1 = len(dataset[dataset['species'] == 1])
    count2 = len(dataset[dataset['species'] == 2])

    weights = [count0, count1, count2]
    return np.argmax(weights)


def regressErr(dataset):
    '''
    输入：数据集
    功能：求数据集划分左右子数据集的误差平方和之和
    输出: 数据集划分后的Gini值
    '''
    #每个种类的几率平方和/划分节点
    count0 = len(dataset[dataset['species'] == 0])
    count1 = len(dataset[dataset['species'] == 1])
    count2 = len(dataset[dataset['species'] == 2])
    count = len(dataset)
    gini = 1 - np.square(np.true_divide(count0, count)) - np.square(np.true_divide(count1, count)) - np.square(np.true_divide(count2, count))
    gini = np.true_divide(gini, 2)
    return gini

def chooseBestSplit(dataset,leafType=regressLeaf,errType=regressErr,threshold=(0.01,4)):
    thresholdErr = threshold[0];thresholdSamples = threshold[1]
    #当数据中输出值都相等时，feature = None,value = 输出值的均值（叶节点）
    if len(set(dataset['Species'].T.tolist())) == 1:
        return None,leafType(dataset)
    Err = errType(dataset)
    bestErr = np.inf; bestFeatureIndex = 0; bestFeatureValue = 0
    featureNames = dataset.columns[0:-1].tolist()
    for featureName in featureNames:
        for featurevalue in dataset[featureName].tolist():
            matLeft,matRight = binarySplitDataSet(dataset,featureName,featurevalue)
            if (np.shape(matLeft)[0] < thresholdSamples) or (np.shape(matRight)[0] < thresholdSamples):
                continue
            temErr = errType(matLeft) + errType(matRight)
            if temErr < bestErr:
                bestErr = temErr
                bestFeatureIndex = featureName
                bestFeatureValue = featurevalue
    #检验在所选出的最优划分特征及其取值下，误差平方和与未划分时的差是否小于阈值，若是，则不适合划分
    if (Err - bestErr) < thresholdErr:
        return None,leafType(dataset)
    matLeft,matRight = binarySplitDataSet(dataset,bestFeatureIndex,bestFeatureValue)
    #检验在所选出的最优划分特征及其取值下，划分的左右数据集的样本数是否小于阈值，若是，则不适合划分
    if (np.shape(matLeft)[0] < thresholdSamples) or (np.shape(matRight)[0] < thresholdSamples):
        return None,leafType(dataset)
    return bestFeatureIndex,bestFeatureValue


def createCARTtree(dataset,leafType=regressLeaf,errType=regressErr,threshold=(1,4)):

    '''
    输入：数据集dataset，叶子节点形式leafType：regressLeaf（回归树）、modelLeaf（模型树）
         损失函数errType:误差平方和也分为regressLeaf和modelLeaf、用户自定义阈值参数：
         误差减少的阈值和子样本集应包含的最少样本个数
    功能：建立回归树或模型树
    输出：以字典嵌套数据形式返回子回归树或子模型树或叶结点
    '''
    feature,value = chooseBestSplit(dataset,leafType,errType,threshold)
    #当不满足阈值或某一子数据集下输出全相等时，返回叶节点
    if feature == None: return value
    returnTree = {}
    leftSet,rightSet = binarySplitDataSet(dataset,feature,value)
    returnTree[feature] = {}
    returnTree[feature]['<=' + str(value) + 'contains' + str(len(leftSet))] = createCARTtree(leftSet,leafType,errType,threshold)
    returnTree[feature]['>' + str(value) + 'contains' + str(len(rightSet))] = createCARTtree(rightSet,leafType,errType,threshold)
    return returnTree
    
if __name__ == '__main__':

    data = pd.read_csv("iris.csv")
    data = data.sample(frac=1.0)
    data = data.reset_index()
    deleteColumns = [0,1]
    data.drop(data.columns[deleteColumns], axis=1, inplace=True)

    trainDataset = data.loc[0:99]
    validationDataset = data.loc[100:129]
    testDataset = data.loc[130:-1]

    cartTree = createCARTtree(trainDataset,threshold=(0.01,4))
    treePlotter.createPlot(cartTree)
