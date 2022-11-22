import time
from mnist import MNIST
import numpy as np
import kmeans as myKMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.datasets import make_blobs


def blobs_setup():
    # let mykmeans and sklearn-kmeans has same initial centorids
    kmeans = KMeans(n_clusters=4, max_iter=1, init='random')
    x, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    kmeans.fit(x)
    initCluster = kmeans.cluster_centers_
    return x, initCluster, 300


def digits_setup_sklearn():
    # let mykmeans and sklearn-kmeans has same initial centorids
    kmeans = KMeans(n_clusters=10, max_iter=1, init='random')
    digits = load_digits()
    kmeans.fit(digits.data)
    initCluster = kmeans.cluster_centers_
    return digits.data, initCluster, 300

def digits_setup_mnist():
    mndata = MNIST('mnist')
    images, labels = mndata.load_training()
    images = np.array(images)
    kmeans = KMeans(n_clusters=10, max_iter=1, init='random')
    kmeans.fit(images)
    initCluster = kmeans.cluster_centers_
    return images, initCluster, 300

def test_seq_blobs():
    x, initCluster, max_iter = blobs_setup()
    kmeans = KMeans(n_clusters=4, max_iter=max_iter, init=initCluster)
    kmeans.fit(x)
    sklearn_loss = kmeans.inertia_
    mykmeans = myKMeans.kmeans64(4, initCluster, max_iter, 1e-4 ,False, False, 8)
    mykmeans.fit(x)
    myloss = mykmeans.inertia_
    assert(abs(sklearn_loss-myloss) / myloss < 1e-10)

def test_digits():
    x, initCluster, max_iter = digits_setup_sklearn()
    kmeans = KMeans(n_clusters=10, max_iter=max_iter, init=initCluster)
    kmeans.fit(x) 
    sklearn_loss = kmeans.inertia_
    mykmeans = myKMeans.kmeans64(10, initCluster, max_iter, 1e-4 ,False, False, 8)
    mykmeans.fit(x)
    myloss = mykmeans.inertia_
    assert(abs(sklearn_loss-myloss) / myloss < 1e-10)

def test_seq_blobs_simd():
    x, initCluster, max_iter = blobs_setup()
    kmeans = KMeans(n_clusters=4, max_iter=max_iter, init=initCluster)
    kmeans.fit(x)
    sklearn_loss = kmeans.inertia_
    mykmeans = myKMeans.kmeans64(4, initCluster, max_iter, 1e-4 ,False, True, 8)
    mykmeans.fit(x)
    myloss = mykmeans.inertia_
    assert(abs(sklearn_loss-myloss) / myloss < 1e-10)

def test_digits_simd():
    x, initCluster, max_iter = digits_setup_sklearn()
    kmeans = KMeans(n_clusters=10, max_iter=max_iter, init=initCluster)
    kmeans.fit(x) 
    sklearn_loss = kmeans.inertia_
    mykmeans = myKMeans.kmeans64(10, initCluster, max_iter, 1e-4 ,False, True, 8)
    mykmeans.fit(x)
    myloss = mykmeans.inertia_
    assert(abs(sklearn_loss-myloss) / myloss < 1e-10)
#
#def test_digits_mnist():
#    x, initCluster, max_iter = digits_setup_mnist()
#    kmeans = KMeans(n_clusters=10, max_iter=max_iter, init=initCluster)
#    kmeans.fit(x) 
#    sklearn_loss = kmeans.inertia_
#    mykmeans = myKMeans.kmeans64(10, initCluster, max_iter, 1e-4 ,False)
#    mykmeans.fit(x)
#    myloss = mykmeans.inertia_
#    assert(abs(sklearn_loss-myloss) / myloss < 1e-10)
