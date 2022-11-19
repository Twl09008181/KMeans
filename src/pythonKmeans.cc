#include <kmeans.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py=pybind11;
using denseArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

dataSetPtr npToCpp(denseArray ndarray){ 
  py::buffer_info info = ndarray.request();
  assert(info.ndim==2);
  dataSetPtr data{info.shape[0],info.shape[1], static_cast<double*>(info.ptr)};
  return data;
}

double kmeansForPy(denseArray ndarray, int k, double epsilon=1e-4, int maxIteration=300, bool verbose=false){
  dataSetPtr data = npToCpp(ndarray);
  return kmeans(data, k, epsilon, maxIteration, verbose);
}
PYBIND11_MODULE(kmeans, m) {
  m.def("seqKmeans", &kmeansForPy,"sequential version of kmeans");
} 
