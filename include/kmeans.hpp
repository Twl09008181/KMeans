#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py=pybind11;

void sequentialKmeans(py::array_t<double>ndarray, int k, double epsilon=1e-4, int maxIteration=300, bool verbose=false);
PYBIND11_MODULE(kmeans, m) {
  m.def("seqKmeans", &sequentialKmeans,"sequential version of kmeans");
} 

#endif
