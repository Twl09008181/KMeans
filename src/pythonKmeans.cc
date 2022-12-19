#include <kmeans.hpp>
#include <alignedAllocator.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace py=pybind11;

template<typename T>
using denseArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template<typename T>
class kmeansForPy{
public:
  kmeansForPy(int k,int maxIter, double tol, bool verbose, bool simd, size_t threadNum)
    :kmeansEngine{k,maxIter,tol,verbose, simd, threadNum}
  {
  }

  // init cluster constructor
  kmeansForPy(int k, denseArray<T> initCluster, int maxIter, double tol, bool verbose, bool simd, size_t threadNum)
    :kmeansForPy{k,maxIter,tol,verbose, simd, threadNum}
  {
    py::buffer_info info = initCluster.request();
    assert(info.ndim==2);
    assert(k==info.shape[0]);
    std::vector<std::vector<T>>initC(k, std::vector<T>(info.shape[1]));
    T* buf = static_cast<T*>(info.ptr);
    for(int i = 0; i < k; i++){
      for(long dim = 0; dim < info.shape[1]; ++dim){
        initC[i][dim] = buf[i*info.shape[1]+dim];
      }
    }
    kmeansEngine._initCluster = initC;
  }
  void fit(denseArray<T> ds){
    dataSetPtr<T> dsptr = npToCpp(ds);
    kmeansEngine.fit(dsptr);
  }
  py::array_t<size_t> predict(denseArray<T>& ds){
    dataSetPtr<T> dsptr = npToCpp(ds);
    auto labels =  kmeansEngine.predict(dsptr);
    // return as np array
    auto nplabels = py::array_t<size_t>(labels.size());
    py::buffer_info buf = nplabels.request();
    size_t *ptr = static_cast<size_t *>(buf.ptr);
    for(size_t i = 0; i < labels.size(); ++i){
      ptr[i] = labels[i];
    }
    return nplabels;
  }
  double inertia(){return kmeansEngine._inertia;};
  py::array_t<T> cluster_centers()
  { 
    auto centers = kmeansEngine._cluster_centers;
    int k = centers.size();
    int dim = centers[0].size();
    // return as np array
    py::array_t<T> npCenters = py::array_t<T>(centers.size()*centers[0].size());
    py::buffer_info buf = npCenters.request();
    T*ptr = static_cast<T *>(buf.ptr);
    for(size_t c = 0; c < k;++c){
      for(size_t d = 0; d < dim; ++d){
        ptr[c*centers[0].size()+d] = centers[c][d];
      }
    }

    npCenters.resize({k, dim});
    return npCenters;
  }
private:
  kmeans<T> kmeansEngine;

  dataSetPtr<T> npToCpp(denseArray<T> ndarray){ 
    py::buffer_info info = ndarray.request();
    assert(info.ndim==2);
    dataSetPtr<T> data{info.shape[0],info.shape[1], static_cast<T*>(info.ptr)};
    return data;
  }
};




PYBIND11_MODULE(kmeans, m) {
  m.doc() = "kmeans";
  pybind11::class_<kmeansForPy<double>>(m, "kmeans64")
      .def(pybind11::init<size_t, size_t, double, bool, bool, size_t>())
      .def(pybind11::init<size_t, denseArray<double>, size_t, double, bool, bool, size_t>())
      .def("fit", &kmeansForPy<double>::fit)
      .def("predict", &kmeansForPy<double>::predict)
      .def_property("inertia_", &kmeansForPy<double>::inertia, nullptr)
      .def_property("cluster_centers_", &kmeansForPy<double>::cluster_centers, nullptr);

  pybind11::class_<kmeansForPy<float>>(m, "kmeans32")
      .def(pybind11::init<size_t, size_t, float, bool, bool, size_t>())
      .def(pybind11::init<size_t, denseArray<float>, size_t, float, bool, bool, size_t>())
      .def("fit", &kmeansForPy<float>::fit)
      .def("predict", &kmeansForPy<float>::predict)
      .def_property("inertia_", &kmeansForPy<float>::inertia, nullptr)
      .def_property("cluster_centers_", &kmeansForPy<float>::cluster_centers, nullptr);
} 
