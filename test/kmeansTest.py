
import time
import numpy as np
import kmeans as myKMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.datasets import make_blobs


def blobs_setup():
    # let mykmeans and sklearn-kmeans has same initial centorids
    initCluster = np.array(
    [[2.35151,0.828001 ],
    [-1.55877,7.24816 ],
    [2.31102,1.30381 ],
    [2.10616,3.49513]])
    max_iter = 300
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    return X, initCluster, max_iter


def digits_setup():
    # let mykmeans and sklearn-kmeans has same initial centorids
    initCluster = np.array(
    [[0,0,0,0,5,11,14,1,0,0,0,10,13,8,15,2,0,0,11,9,4,9,12,0,0,5,16,16,16,16,6,0,0,0,15,16,13,16,3,0,0,0,2,3,1,15,0,0,0,0,0,0,5,5,0,0,0,0,0,0,6,0,0,0],
    [0,3,14,14,16,16,10,0,0,9,15,9,7,1,0,0,0,10,16,11,1,0,0,0,0,1,7,14,9,0,0,0,0,0,0,7,16,0,0,0,0,0,0,6,15,0,0,0,0,1,1,11,10,0,0,0,0,3,15,16,4,0,0,0],
    [0,0,0,1,13,2,0,0,0,0,0,12,14,0,0,0,0,0,6,14,0,0,0,0,0,1,14,5,0,0,0,0,0,9,12,0,12,7,0,0,0,12,14,6,16,14,1,0,0,6,16,16,16,5,0,0,0,0,0,3,14,0,0,0],
    [0,0,1,14,9,0,0,0,0,0,8,13,3,7,1,0,0,1,16,6,5,16,3,0,0,7,13,0,14,11,3,0,0,12,13,5,16,16,9,0,0,13,16,16,15,6,0,0,0,0,3,12,14,0,0,0,0,0,0,15,10,0,0,0],
    [0,0,0,5,11,0,0,0,0,0,1,14,9,0,0,0,0,0,4,14,1,0,0,0,0,0,10,8,0,0,0,0,0,0,13,8,4,6,2,0,0,0,11,16,13,12,13,0,0,0,12,14,4,5,16,2,0,0,1,8,16,13,9,1],
    [0,2,15,16,12,0,0,0,0,8,11,8,16,0,0,0,0,3,1,7,13,0,0,0,0,0,0,10,8,0,0,0,0,0,0,15,5,0,0,0,0,0,7,15,0,0,0,0,0,0,14,11,6,5,2,0,0,1,16,16,16,16,9,0],
    [0,0,2,10,7,0,0,0,0,0,14,16,16,15,1,0,0,4,16,7,3,16,7,0,0,5,16,10,7,16,4,0,0,0,5,14,14,16,4,0,0,0,0,0,0,16,2,0,0,0,4,7,7,16,2,0,0,0,5,12,16,12,0,0],
    [0,0,2,14,13,0,0,0,0,0,14,15,3,0,0,0,0,6,16,2,1,5,0,0,0,10,13,0,5,16,2,0,0,7,16,9,12,16,11,0,0,0,5,12,16,10,2,0,0,0,0,12,12,1,0,0,0,0,0,16,5,0,0,0],
    [0,0,3,8,11,11,1,0,0,0,3,16,16,12,0,0,0,0,2,15,16,12,0,0,0,0,0,16,16,7,0,0,0,0,1,15,16,10,0,0,0,0,1,16,16,6,0,0,0,0,3,16,16,5,0,0,0,0,2,15,16,6,0,0],
    [0,0,0,5,16,9,0,0,0,0,1,13,16,6,0,0,0,0,13,16,16,4,0,0,0,5,15,16,16,5,0,0,0,0,0,10,16,7,0,0,0,0,0,9,16,8,0,0,0,0,0,9,16,13,0,0,0,0,0,5,14,9,0,0]])
    digits = load_digits()
    max_iter = 300
    return digits.data, initCluster, max_iter

def test_seq_blobs():
    X, initCluster, max_iter = blobs_setup()
    kmeans = KMeans(n_clusters=4, max_iter=max_iter, init=initCluster)
    kmeans.fit(X)
    sklearn_loss = kmeans.inertia_
    myloss = myKMeans.seqKmeans(X, 4, 1e-4, max_iter,False)
    assert(abs(sklearn_loss-myloss) / myloss < 1e-10)
def test_digits():
    X, initCluster, max_iter = digits_setup()
    kmeans = KMeans(n_clusters=10, max_iter=max_iter, init=initCluster)
    kmeans.fit(X) 
    sklearn_loss = kmeans.inertia_
    myloss = myKMeans.seqKmeans(X, 10, 1e-4, max_iter, False)
    assert(abs(sklearn_loss-myloss) / myloss < 1e-10)
