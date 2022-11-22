#ifndef KMEANS_HPP
#define KMEANS_HPP
#include <vector>
#include <alignedAllocator.hpp>

class dataSetPtr{
public:
  dataSetPtr(long  n, long  dim, double* buf)
    :_dataNum{n},_dataDim{dim},_dataBuf{buf}
  {}

  long  num()const{return _dataNum;}
  long  dim()const{return _dataDim;}

  double* operator[](unsigned i){
   return _dataBuf + i * _dataDim;
  }
  size_t size()const{return _dataNum;}
  long _dataNum;
  long _dataDim;
  double* _dataBuf;
};

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

  void fit(dataSetPtr& ds);
  int _n_clusters;
  int _maxIter;
  double _tol;
  bool _verbose;
  double _inertia;
  bool _simd;
  size_t _threadNum;
  std::vector<std::vector<double>>_initCluster;
private:

  template<typename dataSet, typename cluster>
  void distanceCaculation(dataSet&, std::vector<cluster>&, long dim, std::vector<std::vector<double>>&out);
  template<typename dataSet, typename cluster>
  std::vector<cluster> updateClusters(dataSet&, const std::vector<std::vector<double>>&distance, long k, long dim);
  template<typename dataSet, typename cluster>
  void fitCore(dataSet& data, std::vector<cluster>&clusters);

  template<typename cluster>
  std::vector<cluster>init(dataSetPtr&ds);
};

#endif
