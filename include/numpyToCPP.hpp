#ifndef NUMPYTOCPP_HPP
#define NUMPYTOCPP_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
namespace py = pybind11;



// use ndtype = np.float64 to avoid copy.(be careful)
void printArray(py::array_t<double>ndarray);

void square(py::array_t<double>ndarray);



PYBIND11_MODULE(numpyToCPP, m) {
    m.def("printArray", &printArray, "print array");
    m.def("square", &square, "square");
} 
#endif
