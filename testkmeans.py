from kmeans import *
from numpy import *
from sklearn.datasets import make_blobs
import pandas as pd

## step 1: load data
print("step 1: load data...")
# dataSet = []
# fileIn = open('testSet.txt')
# for line in fileIn.readlines():
# 	lineArr = line.strip().split('\t')
# 	dataSet.append([float(lineArr[0]), float(lineArr[1])])
# nd = np.genfromtxt("xclara.csv", delimiter=',', skip_header=True)
# dataSet = nd.tolist()

# data = pd.read_csv('xclara.csv')
# data.head()
# f1 = data['V1'].values
# f2 = data['V2'].values
# dataSet = np.array(list(zip(f1, f2)))

n_samples = 3000
random_state = 170
dataSet, target = make_blobs(n_samples=n_samples, random_state=random_state)

## step 2: clustering...
print("step 2: clustering...")
# dataSet = mat(dataSet)
transformation = [[0.60834549, -0.63667341],
                  [-0.40887718, 0.85253229]]
dataSet = dot(dataSet, transformation)
k = 3
centroids, clusterAssment = kmeans(dataSet, k)

## step 3: show the result
print("step 3: show the result...")
showCluster(dataSet, k, centroids, clusterAssment)
