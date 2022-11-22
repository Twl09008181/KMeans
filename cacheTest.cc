#include <iostream>
#include <iomanip>
#include <loadMnist.hpp>
#include <kmeans.hpp>
#include <immintrin.h>
#include <chrono>
#include <cassert>
#include <alignedAllocator.hpp>
using namespace std;

// for alignedVector
double* raw(alignedVector&vec){
  return &vec[0];
}
// for general vector
double* raw(std::vector<double>&vec){
  return &vec[0];
}
// for dataSetPtr's _dataBuf
double* raw(double* p){
  return p;
}

template<typename cluster>
std::vector<cluster>init(dataSetPtr&ds, int _n_clusters){
  long numOfData = ds._dataNum;
  long dim = ds._dataDim;
  std::vector<cluster> clusters(_n_clusters, cluster(ds.dim()));
    srand(0);
  for(long c = 0; c < _n_clusters; ++c){
    int i = rand() % numOfData;
    for(long d = 0; d < dim; d++){
      clusters[c][d] = ds[i][d];
    }
  }
  return clusters;
}


//scalar version 
double squareDistanceScalar(double* p1, double *p2, long s, long e){ 
  double dis = 0;
  for(long b = s; b < e; ++b){ 
    dis += (p1[b]-p2[b]) *(p1[b]-p2[b]);
  }
  return dis;
}
//simd
double squareDistanceSIMD(double *v1, double *v2, long s, long e){

  //mutiple accumulators
  __m256d sumBuffer1 = _mm256_set1_pd(0);
  __m256d sumBuffer2 = _mm256_set1_pd(0);
  __m256d sumBuffer3 = _mm256_set1_pd(0);
  __m256d sumBuffer4 = _mm256_set1_pd(0);
  size_t step = 4;// 4 double = 64 bytes.
  size_t i = s;
  for(;i+4*step < e; i+=4*step){ 
  //for(;i+2*step <= N; i+=2*step){ 
  //for(;i+step <= N; i+=step){ 
    __m256d p1 = _mm256_load_pd(v1+i);
    __m256d p2 = _mm256_load_pd(v2+i);
    __m256d sub1 = _mm256_sub_pd(p1, p2);
    __m256d p3 = _mm256_load_pd(v1+i+4);
    __m256d p4 = _mm256_load_pd(v2+i+4);
    __m256d sub2 = _mm256_sub_pd(p3, p4);
    __m256d p5 = _mm256_load_pd(v1+i+8);
    __m256d p6 = _mm256_load_pd(v2+i+8);
    __m256d sub3 = _mm256_sub_pd(p5, p6);
    __m256d p7 = _mm256_load_pd(v1+i+12);
    __m256d p8 = _mm256_load_pd(v2+i+12);
    __m256d sub4 = _mm256_sub_pd(p7, p8);
    sumBuffer1 = _mm256_fmadd_pd(sub1, sub1, sumBuffer1);
    sumBuffer2 = _mm256_fmadd_pd(sub2, sub2, sumBuffer2);
    sumBuffer3 = _mm256_fmadd_pd(sub3, sub3, sumBuffer3);
    sumBuffer4 = _mm256_fmadd_pd(sub4, sub4, sumBuffer4);
  }

  sumBuffer1 = _mm256_add_pd(sumBuffer1, sumBuffer2);
  sumBuffer2 = _mm256_add_pd(sumBuffer3, sumBuffer4);
  sumBuffer1 = _mm256_add_pd(sumBuffer1, sumBuffer2);

  double result[step];
  _mm256_store_pd(result, sumBuffer1);
  double sum = 0;
  for(int j = 0; j < step; j++)
    sum+=result[j];

  double remain = 0;
  for(;i < e;i++){
    remain += (v1[i]-v2[i])*(v1[i]-v2[i]);
  }
  return sum+ remain;
}


double squareDistance(double *v1, double *v2, long start, long end, bool simd=false){
  bool isAlignment = ((intptr_t(v1) & 0x1f)==0) && ((intptr_t(v2) & 0x1f)==0);
  if(isAlignment && simd){
    return squareDistanceSIMD(v1, v2, start, end);
  }
  else{
    return squareDistanceScalar(v1, v2, start, end);
  }
}

int main(int argc,char*argv[]){
    dataset mnist = load_mnist("./mnist/train-images-idx3-ubyte","./mnist/train-labels-idx1-ubyte"); 
    //showMnist(mnist);
    
    dataSetPtr ds(mnist.num, mnist.dim, &mnist.data[0]);
    long k = 10;
    long numOfData = ds.size();
    long dim = ds.dim();

    using dataType = alignedVector;
    std::vector<dataType>data(ds.size(), alignedVector(ds.dim()));
    for(long i = 0; i < ds.size(); i++)
      for(long d = 0; d < ds.dim(); ++d) 
        data[i][d] = ds[i][d];
    std::vector<dataType>clusters = init<dataType>(ds, k);


    std::vector<std::vector<double>>distance(numOfData, std::vector<double>(k,0));
    for(long i = 0; i < ds.size(); i++){ 
        for(int c = 0; c < k;++c){ 
            distance[i][c] = squareDistance(raw(ds[i]), raw(clusters[c]), 0, dim, true);
        }
    }


   return 0;
}
