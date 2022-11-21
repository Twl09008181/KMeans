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
  long _dataNum;
  long _dataDim;
  double* _dataBuf;
};

class kmeans{
public:
  kmeans(int n_clusters, int maxIter=300, double tol=1e-4, bool verbose=false)
    :
    _n_clusters{n_clusters},
    _maxIter{maxIter},
    _tol{tol},
    _verbose{verbose},
    _inertia{-1}
  {}

  void fit(dataSetPtr& ds);
  int _n_clusters;
  int _maxIter;
  double _tol;
  bool _verbose;
  double _inertia;
  std::vector<alignedVector>_initCluster;
private:
  //std::vector<std::vector<double>>init(dataSetPtr&ds);
  std::vector<alignedVector>init(std::vector<alignedVector>&ds);
};

#endif
