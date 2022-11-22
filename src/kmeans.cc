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
#include <iterator>


void addVecSIMD(double* v1, double *v2, long N){ 
  size_t step = 4;// 4 double = 64 bytes.
  size_t i = 0;
  for(;i+step <= N; i+=step){ 
    __m256d p1 = _mm256_load_pd(v1+i);
    __m256d p2 = _mm256_load_pd(v2+i);
    __m256d addBuffer = _mm256_add_pd(p1, p2);
    _mm256_store_pd(v1+i, addBuffer);
  }
  for(;i < N;i++){
    v1[i] += v2[i];
  }
}

void addVecScalar(double* p1, double *p2, long dim){ 
  for(long b = 0; b < dim; ++b){ 
    p1[b]+=p2[b];
  }
}


void addVec(double* v1, double *v2, long N, bool simd=false){ 
  bool isAlignment = ((intptr_t(v1) & 0x1f)==0) && ((intptr_t(v2) & 0x1f)==0);
  if(isAlignment && simd){
     addVecSIMD(v1, v2, N);
  }
  else{
     addVecScalar(v1, v2, N);
  }
}


void scaling(double* p1, double scale, long dim){ 
  for(long b = 0; b < dim; ++b){ 
    p1[b] /= scale;
  }
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
std::vector<cluster>kmeans::init(dataSetPtr&ds){

  long numOfData = ds._dataNum;
  long dim = ds._dataDim;

  //random initialization
  if(_initCluster.empty()){
    srand(0);
    _initCluster.resize(_n_clusters, std::vector<double>(dim));
    for(long c = 0; c < _n_clusters; ++c){
      int i = rand() % numOfData;
      if(_verbose){std::cout<<"\n Initialization cluster "<<c<<" : \n";}
      for(long d = 0; d < dim; d++){
        _initCluster[c][d] = ds[i][d];
        if(_verbose){std::cout<<_initCluster[c][d]<<" ";}
      }
    }
  }

  std::vector<cluster> clusters(_initCluster.size(), cluster( _initCluster.at(0).size()));
  for(long c = 0; c < _n_clusters; ++c){
    for(long d = 0; d < dim; d++){
      clusters[c][d] = _initCluster[c][d];
    }
  }
  return clusters;
}




template<typename dataSet, typename cluster>
void kmeans::distanceCaculation(
    dataSet& ds,
    std::vector<cluster>& clusters,
    long dim,
    std::vector<std::vector<double>>&distance)
{ 
  #pragma omp parallel for  num_threads(_threadNum)
  for(long i = 0; i < ds.size(); i++){ 
      for(int c = 0; c < clusters.size();++c){ 
          distance[i][c] = squareDistance(raw(ds[i]), raw(clusters[c]), 0, dim, _simd);
      }
  }
}


template<typename dataSet, typename cluster>
std::vector<cluster> kmeans::updateClusters(
    dataSet& data,
    const std::vector<std::vector<double>>&distances,
    long k,
    long dim)
{

    //assign data to closet cluster.
    struct collector{
      std::vector<cluster>cSum;
      std::vector<long>cSize;
    };

    collector collectors[_threadNum];
    for(int i = 0; i < _threadNum; i++){
      collectors[i].cSum.resize(k, cluster(dim,0));
      collectors[i].cSize.resize(k,0);
    }

    _inertia = 0;
    #pragma omp parallel num_threads(_threadNum)
    {
      int id = omp_get_thread_num();
      for(long i = id; i < data.size(); i+=_threadNum){
        long closet=0;
        double minimumDist = std::numeric_limits<double>::max();
        for(long c = 0; c < k; c++){
          if(distances[i][c] < minimumDist){
            minimumDist = distances[i][c];
            closet = c;
          }
        }
        addVec(raw(collectors[id].cSum[closet]), raw(data[i]), dim, _simd);
        collectors[id].cSize[closet]++;
        #pragma omp atomic
        _inertia += minimumDist;
      }
    }

    //combine
    for(long i = 1; i < _threadNum; i++){
      for(long c = 0; c < k; c++){
        addVec(raw(collectors[0].cSum[c]), raw(collectors[i].cSum[c]), dim);
        collectors[0].cSize[c] += collectors[i].cSize[c];
      }
    }
    //update clusters
    for(long c = 0; c < k; c++){
      scaling(raw(collectors[0].cSum[c]), collectors[0].cSize[c], dim);
    }
    return std::move(collectors[0].cSum);
}



void kmeans::fit(dataSetPtr& ds){
  if(_verbose){
    std::cout<<"num of data:"<<ds.size()<<"\n";
    std::cout<<"num of dim:"<<ds.dim()<<"\n";
  }
  if(_simd){
    // make sure data is aligned.
    using dataType = alignedVector;
    std::vector<dataType>data(ds.size(), alignedVector(ds.dim()));
    for(long i = 0; i < ds.size(); i++)
      for(long d = 0; d < ds.dim(); ++d) 
        data[i][d] = ds[i][d];
    std::vector<dataType> clusters = init<dataType>(ds);
    fitCore(data, clusters);
  }
  else{
    using cluster = std::vector<double>;
    std::vector<cluster> clusters = init<cluster>(ds);
    fitCore(ds, clusters);
  }
}

template<typename dataSet, typename cluster>
void kmeans::fitCore(dataSet& data, std::vector<cluster>&clusters){ 

  //basic information
  long k = _n_clusters;
  long numOfData = data.size();
  long dim = clusters.at(0).size();

  // iteration 
  double difference = std::numeric_limits<double>::max();
  int iters = 0;

  // distance array
  std::vector<std::vector<double>>squareDistances(numOfData, std::vector<double>(k,0));

  // core schedule
  while(difference > _tol && iters<_maxIter){

    // step1 
    distanceCaculation(data, clusters, dim, squareDistances);

    // step2
    auto nextClusters = updateClusters<dataSet, cluster>(data, squareDistances, k, dim);

    // check convergence
    difference = 0;
    for(long c = 0; c < k; c++){
      difference += squareDistance(raw(nextClusters[c]), raw(clusters[c]), 0, dim);
    }
    difference = sqrt(difference);

    if(_verbose){
      std::cout<<"\nIteration "<<iters<<", inertia "<<_inertia<<".\n";
    }

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

