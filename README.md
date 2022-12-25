# KMeans implementation

## Basic Information

KMeans is a famous algorithm for clustering, it is easy to implmenet and use.        
It is widely used in data analysis and machine learning sphere.   
A lot of algorithm is based on this simple and classical algorithm, like Spectral Clustering or data Preprocessing for reducing the complexity for latter training.   
Generally, you can use KMeans when you want to group some data with similar attributes.   

## Usage



### interface


c++
```
// data set ptr

using T = double;
// prepare data
dataset<T> mnist = load_mnist<T>("./mnist/train-images-idx3-ubyte","./mnist/train-labels-idx1-ubyte"); 
dataSetPtr<T> mnistPtr(mnist.num, mnist.dim, &mnist.data[0]);


// kmeans setting
int threadNum = 8;
size_t k = 10;
bool simd = false;
bool verbose = true;
int maxIter = 300;
double tol = 1e-4;
kmeans<T> kms(k, maxIter, tol, verbose, simd, threadNum);


// fit 
kms.fit(mnistPtr);

// get center
auto centers = kms._cluster_centers;
// prediction
auto labels = kms.predict(mnistPtr);
```


python
```
$PYTHONPATH=build python3 example/python/example.py
```

```
# import module
import kmeans as myKMeans

# prepare data
data, labels = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = data.shape, np.unique(labels).size
reduced_data = PCA(n_components=2).fit_transform(data)

# kmeans setting
kmeans = myKMeans.kmeans64(10,300,1e-4,False,True,16)

# fit
kmeans.fit(reduced_data)

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# predict
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

```

### example

check example/


c++
```
$ make c++example
```

python
```
$ make pyexample
```



