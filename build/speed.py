from time import perf_counter
from mnist import MNIST
import numpy as np
import kmeans as myKMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.datasets import make_blobs
import sys


def digits_setup_sklearn():
    # let mykmeans and sklearn-kmeans has same initial centorids
    kmeans = KMeans(n_clusters=10, max_iter=1, init='random')
    digits = load_digits()
    kmeans.fit(digits.data)
    initCluster = kmeans.cluster_centers_
    return digits.data, initCluster, 300

def digits_setup_mnist64():
    mndata = MNIST('mnist')
    images, labels = mndata.load_training()
    images = np.array(images, dtype=np.float64)
    kmeans = KMeans(n_clusters=10, max_iter=1, init='random')
    kmeans.fit(images)
    initCluster = kmeans.cluster_centers_
    initCluster = np.array(kmeans.cluster_centers_, dtype=np.float64)
    return images, initCluster, 300

def digits_setup_mnist32():
    mndata = MNIST('mnist')
    images, labels = mndata.load_training()
    images = np.array(images, dtype=np.float32)
    kmeans = KMeans(n_clusters=10, max_iter=1, init='random')
    kmeans.fit(images)
    initCluster = np.array(kmeans.cluster_centers_, dtype=np.float32)
    return images, initCluster, 300


def test_digits_mnist(engine, dataType, x, initCluster, threadNum):

    if(engine == "sklearn"):
        kmeans = KMeans(n_clusters=10, max_iter=max_iter, init=initCluster, n_init=1)
    
    elif(engine =="simd"):
        if dataType == 32:
            kmeans = myKMeans.kmeans32(10, initCluster, max_iter, 1e-4 ,False, True, threadNum)
        else:
            kmeans = myKMeans.kmeans64(10, initCluster, max_iter, 1e-4 ,False, True, threadNum)
    else: 
        if dataType == 32:
            kmeans = myKMeans.kmeans32(10, initCluster, max_iter, 1e-4 ,False, False, threadNum)
        else:
            kmeans = myKMeans.kmeans64(10, initCluster, max_iter, 1e-4 ,False, False, threadNum)

    s = perf_counter()
    kmeans.fit(x)
    e = perf_counter()
    return e-s

def warn(*arg, **kwargs):
    pass
import warnings
warnings.warn = warn

engine, dataType, repeat, threadNum = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
if dataType==32:
    x, initCluster, max_iter = digits_setup_mnist32()
elif dataType == 64:
    x, initCluster, max_iter = digits_setup_mnist64()
else:
    print("not support type")


#repeat
t = 0
for i in range(repeat):
    t+=test_digits_mnist(engine, dataType, x, initCluster, threadNum)
print(engine,dataType,t/repeat,"s (avg)")
