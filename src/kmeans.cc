#include <kmeans.hpp>
#include <numeric>
#include <limits>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <vector>
#include <math.h>
#include <immintrin.h>
#include <alignedAllocator.hpp>


void addVecSIMD(double* v1, double *v2, long N){ 
  size_t step = 4;// 4 double = 64 bytes.
  size_t i = 0;
  for(;i+step <= N; i+=step){ 
    __m256d p1 = _mm256_load_pd(v1+i);
    __m256d p2 = _mm256_load_pd(v2+i);
    __m256d addBuffer = _mm256_add_pd(p1, p2);
    _mm256_store_pd(v1+i, addBuffer);
  }
}

void addVec(double* p1, double *p2, long dim){ 
  for(long b = 0; b < dim; ++b){ 
    p1[b]+=p2[b];
  }
}

void scaling(double* p1, double scale, long dim){ 
  for(long b = 0; b < dim; ++b){ 
    p1[b] /= scale;
  }
}



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


//simd future?
double squareDistance(double* p1, double *p2, long s, long e){ 
  double dis = 0;
  for(long b = s; b < e; ++b){ 
    dis += (p1[b]-p2[b]) *(p1[b]-p2[b]);
  }
  return dis;
}

double* raw(std::vector<double>&vec){
  return &vec[0];
}

double* raw(alignedVector&vec){
  return &vec[0];
}
//
//std::vector<std::vector<double>>kmeans::init(dataSetPtr&ds){
//  if(_initCluster.empty()){
//    srand(0);
//    long numOfData = ds._dataNum;
//    long dim = ds._dataDim;
//    _initCluster.resize(_n_clusters, std::vector<double>(dim));
//    for(long c = 0; c < _n_clusters; ++c){
//      int i = rand() % numOfData;
//      if(_verbose){
//        std::cout<<"\n Initialization cluster "<<c<<" : \n";
//      }
//      for(long d = 0; d < dim; d++){
//        _initCluster[c][d] = ds[i][d];
//        if(_verbose){
//          std::cout<<_initCluster[c][d]<<" ";
//        }
//      }
//    }
//  }
//  return _initCluster;
//}


std::vector<alignedVector> kmeans::init(std::vector<alignedVector>&ds){ 
  if(_initCluster.empty()){
    srand(0);
    long numOfData = ds.size();
    long dim = ds[0].size();
    _initCluster.resize(_n_clusters, alignedVector(dim));
    for(long c = 0; c < _n_clusters; ++c){
      int i = rand() % numOfData;
      if(_verbose){
        std::cout<<"\n Initialization cluster "<<c<<" : \n";
      }
      for(long d = 0; d < dim; d++){
        _initCluster[c][d] = ds[i][d];
        if(_verbose){
          std::cout<<_initCluster[c][d]<<" ";
        }
      }
    }
  }
  return _initCluster;
}


//void kmeans::fit(dataSetPtr& data){ 
//
//
//  long k = _n_clusters;
//  long numOfData = data._dataNum;
//  long dim = data._dataDim;
//  if(_verbose){
//    std::cout<<"num of data:"<<numOfData<<"\n";
//    std::cout<<"num of dim:"<<dim<<"\n";
//  }
//  //init
//  std::vector<std::vector<double>>clusters = init(data);
//  double difference = std::numeric_limits<double>::max();
//  int iters = 0;
//  std::vector<std::vector<double>>squareDistances(numOfData, std::vector<double>(k,0));
//  while(difference > _tol && iters<_maxIter){
//
//
//    #pragma omp parallel for
//    for(long i = 0; i < numOfData; i++){ 
//      for(int c = 0; c < k;++c){ 
//          squareDistances[i][c] = squareDistance(data[i], raw(clusters[c]), 0, dim);
//      }
//    }
//
//    std::vector<std::vector<double>>nextClusters(k, std::vector<double>(dim,0));
//    std::vector<long>clusterSize(k,0);
//    _inertia = 0;
//    //assign data to closet cluster.
//    for(long i = 0; i < numOfData; i++){
//      long closet=0;
//      double minimumDist = std::numeric_limits<double>::max();
//      for(long c = 0; c < k; c++){
//        if(squareDistances[i][c] < minimumDist){
//          minimumDist = squareDistances[i][c];
//          closet = c;
//        }
//      }
//      addVec(raw(nextClusters[closet]), data[i], dim);
//      clusterSize[closet]++;
//      _inertia += minimumDist;
//    }
//
//
//
//    if(_verbose){
//      std::cout<<"\nIteration "<<iters<<", inertia "<<_inertia<<".\n";
//    }
//
//    //update clusters
//    for(long c = 0; c < k; c++){
//      scaling(raw(nextClusters[c]), clusterSize[c], dim);
//    }
//
//    difference = 0;
//    for(long c = 0; c < k; c++){
//      difference += squareDistance(raw(nextClusters[c]), raw(clusters[c]), 0,dim);
//    }
//    difference = sqrt(difference);
//    
//    clusters = std::move(nextClusters);
//    iters++;
//  }
//
//  if(_verbose){
//    std::cout<<"Iterations:"<<iters<<"\n";
//    for(long c = 0; c < k; ++c){
//      std::cout<<"cluster"<<c<<":\n";
//      for(long d = 0; d < dim; d++)
//        std::cout<<clusters[c][d]<<" ";
//      std::cout<<"\n";
//    }
//    std::cout<<"inertia:"<<_inertia<<"\n";
//
//  }
//}
//



void kmeans::fit(dataSetPtr& ds){ 

  long k = _n_clusters;
  long numOfData = ds._dataNum;
  long dim = ds._dataDim;

  std::vector<alignedVector>data(numOfData, alignedVector(dim));

  for(long i = 0; i < numOfData; i++){ 
    for(long d = 0; d < dim; ++d){ 
      data[i][d] = ds[i][d];
    }
  }

  if(_verbose){
    std::cout<<"num of data:"<<numOfData<<"\n";
    std::cout<<"num of dim:"<<dim<<"\n";
  }
  //init
  std::vector<alignedVector>clusters = init(data);
  double difference = std::numeric_limits<double>::max();
  int iters = 0;
  std::vector<std::vector<double>>squareDistances(numOfData, std::vector<double>(k,0));
  while(difference > _tol && iters<_maxIter){


    //#pragma omp parallel for
    for(long i = 0; i < numOfData; i++){ 
      for(int c = 0; c < k;++c){ 
          squareDistances[i][c] = squareDistanceSIMD(raw(data[i]), raw(clusters[c]), dim);
          //squareDistances[i][c] = squareDistance(raw(data[i]), raw(clusters[c]), 0, dim);
      }
    }

    std::vector<alignedVector>nextClusters(k, alignedVector(dim,0));
    std::vector<long>clusterSize(k,0);
    _inertia = 0;
    //assign data to closet cluster.
    for(long i = 0; i < numOfData; i++){
      long closet=0;
      double minimumDist = std::numeric_limits<double>::max();
      for(long c = 0; c < k; c++){
        if(squareDistances[i][c] < minimumDist){
          minimumDist = squareDistances[i][c];
          closet = c;
        }
      }
      addVecSIMD(raw(nextClusters[closet]), raw(data[i]), dim);
      clusterSize[closet]++;
      _inertia += minimumDist;
    }



    if(_verbose){
      std::cout<<"\nIteration "<<iters<<", inertia "<<_inertia<<".\n";
    }

    //update clusters
    for(long c = 0; c < k; c++){
      scaling(raw(nextClusters[c]), clusterSize[c], dim);
    }

    difference = 0;
    for(long c = 0; c < k; c++){
      difference += squareDistance(raw(nextClusters[c]), raw(clusters[c]), 0,dim);
    }
    difference = sqrt(difference);
    
    clusters = std::move(nextClusters);
    iters++;
  }

  if(_verbose){
    std::cout<<"Iterations:"<<iters<<"\n";
    for(long c = 0; c < k; ++c){
      std::cout<<"cluster"<<c<<":\n";
      for(long d = 0; d < dim; d++)
        std::cout<<clusters[c][d]<<" ";
      std::cout<<"\n";
    }
    std::cout<<"inertia:"<<_inertia<<"\n";

  }
}
