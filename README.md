# KMeans implementation

## Basic Information

KMeans is a famous algorithm for clustering, it is easy to implmenet and use.        
It is widely used in data analysis and machine learning sphere.   
A lot of algorithm is based on this simple and classical algorithm, like Spectral Clustering or data Preprocessing for reducing the complexity for latter training.   
Generally, you can use KMeans when you want to group some data with similar attributes.   


## Problem to solve 

Kmeans pseudo code:

```
K-means(D,k,epsilon){
randomly initilize k Centroids vector :u1,u2,..uk

repeat until convergence:
  set Centroids set empty for C1,C2,..Ck.
  
  //assign to Centroid 
  for each data in D:
    caculate the distance between data and Centroids(u1,u2,...uk)
    assign data to the closest centroids Cj.
  
  //update Centroid vector 
  foreach Cj in Centroids:
    uj = 0
    for each data in Cj:
      uj += data
    uj /= |Cj|

output the Centroids.
}

``` 

KMeans is simple but very computationally intensive,    
It can be splitted into two major steps:    
1. Assign Step : to cacaulate distance between data and all centroids.    
2. Centroid Updating Step: to update the coordinate of each centroid according to assignment step.      

I think it's worthy to speed up kmeans by C++, and make it a general usage for python programmer.     
The interface of kmeans is simple, we can just use 2D array to represent the dataset.     
It's easy to get dataset from sklearn.    


## Goal

1. Implment a C++ version KMeans.
2. Try to speed up with parallel programming skill.
3. Use pybind to make it callable by python.
4. Comparing the efficiency with the sklearn kmeans model.


## Prospective Users
C++ or python programmer who want use k-means as a step to build their algorithm.

## System Architecture


API

```
//group to K-centroid, the return value in range [0,K-1] to indicate which group it belong to.
vector<int> kmeans(vector<vector<float>>&dataset, int K, double epsilone);
```

1. C++/Python
2. Pybind
3. handcraft thread pool or OpenMP.

## Engineering Infrastructure     

Build System : make       
Version Control : git   
Test : pytest or goolge test      


## Schedule   

week0(11/01): sequential kmeans algorithm study.          
week1(11/07): Implement sequential kmeans by C++ and try to use pybind to call by python.   
week2(11/14): parallel kmeans algorithm study.      
week3(11/21): parallel programming framework study.     
week4(11/28): Implement paralle kmeans by C++.      
week5(12/05): Testing and compare with sklearn kmeans.      
week6(12/12): Buffer time for checking speedup or add another features.     
week7(12/19): Prepare slides.     
week8(12/26): Project Present     
  
## References
Parallel K-Means Clustering Based on MapReduce:https://link.springer.com/chapter/10.1007/978-3-642-10665-1_71


## Daily note

### 11/12
The first thing we need is to understand how to pass numpy-data to C++ and avoid copy cost,
I refer to https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html and find an usefull protocol : buffer protocol.


we can use py::array_t<double>  to get the numpy array and use arrayName.request() to get the information.
and use info.ndim , info.shape[0],info.shape[1]...info.shape[info.ndim-1] to get shapes and info.ptr to get buffer,
then we can use double ptr to get data from info.ptr.

for example, I write following code for test.

src/numpyToCpp.cpp
```
#include "../include/numpyToCPP.hpp"
// use ndtype = np.float64 to avoid copy.(be careful)
void printArray(py::array_t<double>ndarray){
  py::buffer_info info = ndarray.request();
  std::cout<<"ndim:"<<info.ndim<<"\n";
  size_t size = 1;
  for(int dim = 0; dim < info.ndim; dim++){
    std::cout<<"shape["<<dim<<"]="<<info.shape[dim]<<"\n";
    size *= info.shape[dim];
  }
  double *ptr = static_cast<double*>(info.ptr);
  for(size_t i = 0; i < size; i++){
    std::cout<<*(ptr+i)<<" ";
  }
  std::cout<<"\n";
}

void square(py::array_t<double>ndarray){
  py::buffer_info info = ndarray.request();
  size_t size = 1;
  for(int dim = 0; dim < info.ndim; dim++){
    size *= info.shape[dim];
  }
  double *ptr = static_cast<double*>(info.ptr);
  for(size_t i = 0; i < size; i++){
    *(ptr+i) = *(ptr+i) * *(ptr+i);
  }
}

```

include/numpyToCpp.hpp
```
#ifndef NUMPYTOCPP_HPP
#define NUMPYTOCPP_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
namespace py = pybind11;


// use ndtype = np.float64 to avoid copy.(be careful)
void printArray(py::array_t<double>ndarray);

void square(py::array_t<double>ndarray);



PYBIND11_MODULE(numpyToCPP, m) {
    m.def("printArray", &printArray, "print array");
    m.def("square", &square, "square");
} 
#endif

```

and use blobs to test my kmeans, it works!.


### 11/13 

I test the handwritten-digits dataset and find bug.   
The bug is:   
the result of my kmeans differ to the sklearn version.(the final centroids)   
    
It cost me a lot time to find the reason cause the bug because the bug can happen due to a lot reason:    
sklearn kmeans has a lot of input-args and may affect the result, like: n_init, init, algorithm...    

I try to set init-centroids are same in both my version and sk-learn, and check the algorithm sklearn use is same as mine,    
even when the k = 1, the result is wrong, it's easy to say when k = 1, the cluster will be average of all data, howerver my version is not the case.    
    
so i finally find that the numpy data pass into my c++ program are not same!      
And I find the description:   
```
Data in NumPy arrays is not guaranteed to packed in a dense manner; furthermore, entries can be separated by arbitrary column and row strides. Sometimes, it can be 
useful to require a function to only accept dense arrays using either the C (row-major) or Fortran (column-major) ordering. This can be accomplished via a second   
template argument with values py::array::c_style or py::array::f_style.   

``` 
    
so I find that I need to restrict the input to only dense-numpy-array.    
        
I fix following interface           
```
namespace py=pybind11;        
using denseArray = py::array_t<double, py::array::c_style | py::array::forcecast>;      
double sequentialKmeans(denseArray ndarray, int k, double epsilon=1e-4, int maxIteration=300, bool verbose=false);      
PYBIND11_MODULE(kmeans, m) {    
  m.def("seqKmeans", &sequentialKmeans,"sequential version of kmeans");   
} 
```

and it works now, I also write the test-case in test/kmeansTest.py      


### 11/19

I update the interface so I can use the same inital cluster to test the functionality.
I also write a C++ version dataloader to test c++ code and performance,
Finally, I find the computation cost of cacaulating distance between cluster and data are very heavy.
![nosimd](https://user-images.githubusercontent.com/52790122/202999188-f0eddac0-234f-46fb-bfd3-7fa20920050a.PNG)  

### 11/20
I study the SIMD usage for speeding the distance caculation, and I study following articles.


1. SIMD introduction 
  https://stackoverflow.blog/2020/07/08/improving-performance-with-simd-intrinsics-in-three-use-cases/
2. SIMD need alignment
  https://xhy3054.github.io/memory-alignment/
3. what is memory-alignment  
  http://opass.logdown.com/posts/743054-about-memory-alignment  
  https://medium.com/starbugs/illustrate-how-data-alignment-affects-memory-usage-d29bf9d5bf08
  https://stackoverflow.com/questions/10224564/what-does-alignment-to-16-byte-boundary-mean-in-x86
4. API
  https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html# 

After study these articles, I try to write my SIMD-version distanceSquare cacaulation.

### 11/21

I use custom allocator to ensure the memory std::vector use is aligned.
```
template <class T>
struct alignedAllocator
{
  using value_type = T;

  T* allocate(std::size_t n){
    T *p = static_cast<T*>(aligned_alloc(32, (n*sizeof(double)/32+1)*32));
    if(p)
      return p;
    else
      throw std::bad_alloc();
  }

  void deallocate(T* p, std::size_t n){
    free(p);
  }
};

```

and write quareDistanceSIMD   

```

//simd
double squareDistanceSIMD(double *v1, double *v2, long N){

  __m256d sumBuffer = _mm256_set1_pd(0);
  size_t step = 4;// 4 double = 64 bytes.
  size_t i = 0;
  for(;i+step <= N; i+=step){ 
    __m256d p1 = _mm256_load_pd(v1+i);
    __m256d p2 = _mm256_load_pd(v2+i);
    __m256d sub1 = _mm256_sub_pd(p1, p2);
    __m256d sub2 = _mm256_sub_pd(p1, p2);
    sumBuffer = _mm256_add_pd(sumBuffer, _mm256_mul_pd(sub1, sub2));
  }

  double result[step];
  _mm256_store_pd(result, sumBuffer);
  double remain = 0;
  for(;i < N;i++){
    remain += (v1[i]-v2[i])*(v1[i]-v2[i]);
  }
  double s = 0;
  for(int j = 0; j < step; j++)
    s+=result[j];
  return s + remain;
}
``` 


testing result:


python test:

testcase1: data shape: 1797x64  (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)     
sklearn: 0.007936522000818513s    
non-simd 0.011200033000932308s      
simd 0.009062467997864587s    

testcase2: data shape: 60000x784  (http://yann.lecun.com/exdb/mnist/)     
sklearn: 6.083494722999603s    
non-simd 42.197882291002315s      
simd 18.067461017002643s      
    
c++ test:(different initial cluster from above)     
    
data shape: 60000x784       
non-simd : 16823ms    
simd :5859ms    
    
It seems simd can really help to speed up, however, sklearn version is very strong in big-case.   
I need to find some other optimization way in the future.   








