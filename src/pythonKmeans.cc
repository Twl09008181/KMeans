#include <kmeans.hpp>
#include <alignedAllocator.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py=pybind11;
using denseArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

class kmeansForPy{
public:
  kmeansForPy(int k,int maxIter, double tol, bool verbose, bool simd)
    :kmeansEngine{k,maxIter,tol,verbose, simd}
  {
  }

  // init cluster constructor
  kmeansForPy(int k, denseArray initCluster, int maxIter, double tol, bool verbose, bool simd)
    :kmeansForPy{k,maxIter,tol,verbose, simd}
  {
    py::buffer_info info = initCluster.request();
    assert(info.ndim==2);
    assert(k==info.shape[0]);
    std::vector<std::vector<double>>initC(k, std::vector<double>(info.shape[1]));
    double* buf = static_cast<double*>(info.ptr);
    for(int i = 0; i < k; i++){
      for(long dim = 0; dim < info.shape[1]; ++dim){
        initC[i][dim] = buf[i*info.shape[1]+dim];
      }
    }
    kmeansEngine._initCluster = initC;
  }
  void fit(denseArray ds){
    dataSetPtr dsptr = npToCpp(ds);
    kmeansEngine.fit(dsptr);
  }
  double inertia(){return kmeansEngine._inertia;};
private:
  kmeans kmeansEngine;

  dataSetPtr npToCpp(denseArray ndarray){ 
    py::buffer_info info = ndarray.request();
    assert(info.ndim==2);
    dataSetPtr data{info.shape[0],info.shape[1], static_cast<double*>(info.ptr)};
    return data;
  }
};




PYBIND11_MODULE(kmeans, m) {
  m.doc() = "kmeans";
  pybind11::class_<kmeansForPy>(m, "kmeans")
      .def(pybind11::init<size_t, size_t, double, bool, bool>())
      .def(pybind11::init<size_t, denseArray, size_t, double, bool, bool>())
      .def("fit", &kmeansForPy::fit).
      def_property("inertia_", &kmeansForPy::inertia, nullptr);
} 
