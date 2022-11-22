#ifndef KMEANS_HPP
#define KMEANS_HPP
#include <vector>
#include <alignedAllocator.hpp>
#include <iostream>
#include <omp.h>
#include <numeric>
#include <limits>
#include <math.h>

template<typename T>
class dataSetPtr{
public:
  dataSetPtr(long  n, long  dim, T* buf)
    :_dataNum{n},_dataDim{dim},_dataBuf{buf}
  {}

  long  num()const{return _dataNum;}
  long  dim()const{return _dataDim;}

  T* operator[](unsigned i){
   return _dataBuf + i * _dataDim;
  }
  size_t size()const{return _dataNum;}
  long _dataNum;
  long _dataDim;
  T* _dataBuf;
};


template<typename T>
class kmeans{
public:
  kmeans(int n_clusters, int maxIter=300, double tol=1e-4, bool verbose=false, bool simd=false, size_t threadNum=4)
    :
    _n_clusters{n_clusters},
    _maxIter{maxIter},
    _tol{tol},
    _verbose{verbose},
    _inertia{-1},
    _simd{simd},
    _threadNum{threadNum}
  {}

  using alignedVector = alignedVectorType<T>;
  void fit(dataSetPtr<T>& ds);
  int _n_clusters;
  int _maxIter;
  double _tol;
  bool _verbose;
  double _inertia;
  bool _simd;
  size_t _threadNum;
  std::vector<std::vector<T>>_initCluster;
private:
  template<typename dataSet, typename cluster>
  void distanceCaculation(dataSet&, std::vector<cluster>&, long dim, std::vector<std::vector<double>>&out);
  template<typename dataSet, typename cluster>
  std::vector<cluster> updateClusters(dataSet&, const std::vector<std::vector<double>>&distance, long k, long dim);
  template<typename dataSet, typename cluster>
  void fitCore(dataSet& data, std::vector<cluster>&clusters);

  template<typename cluster>
  std::vector<cluster>init(dataSetPtr<T>&ds);
};


// for alignedVector
template<typename T>
T* raw(alignedVectorType<T>&vec){
  return &vec[0];
}
// for general vector
template<typename T>
T* raw(std::vector<T>&vec){
  return &vec[0];
}
// for dataSetPtr's _dataBuf
template<typename T>
T* raw(T* p){
  return p;
}

double squareDistance(double *v1, double *v2, long start, long end, bool simd=false);
double squareDistance(float *v1, float *v2, long start, long end, bool simd=false);

void addVec(double* v1, double *v2, long N, bool simd=false);
void addVec(float* v1, float *v2, long N, bool simd=false);
template<typename T>
void scaling(T* p1, long scale, long dim){ 
  for(long b = 0; b < dim; ++b){ 
    p1[b] /= scale;
  }
}

template<typename T>
template<typename cluster>
std::vector<cluster>kmeans<T>::init(dataSetPtr<T>&ds){

  long numOfData = ds._dataNum;
  long dim = ds._dataDim;

  //random initialization
  if(_initCluster.empty()){
    srand(0);
    _initCluster.resize(_n_clusters, std::vector<T>(dim));
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



template<typename T>
template<typename dataSet, typename cluster>
void kmeans<T>::distanceCaculation(
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


template<typename T>
template<typename dataSet, typename cluster>
std::vector<cluster> kmeans<T>::updateClusters(
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
    #pragma omp parallel num_threads(_threadNum) reduction(+:_inertia)
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



template<typename T>
void kmeans<T>::fit(dataSetPtr<T>& ds){
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
    using cluster = std::vector<T>;
    std::vector<cluster> clusters = init<cluster>(ds);
    fitCore(ds, clusters);
  }
}

template<typename T>
template<typename dataSet, typename cluster>
void kmeans<T>::fitCore(dataSet& data, std::vector<cluster>&clusters){ 

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



#endif
