#ifndef KMEANS_HPP
#define KMEANS_HPP

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


double kmeans(dataSetPtr& ds, int k, double epsilon=1e-4, int maxIter=300, bool verbose=false);

#endif
