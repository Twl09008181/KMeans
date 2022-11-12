#ifndef NUMPYTOCPP_HPP
#define NUMPYTOCPP_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
namespace py = pybind11;


class array2d{
public:
  array2d(size_t row, size_t col, double*buf)
    :_nrow{row},_ncol{col},_buf{buf}{}
  size_t nrow()const{return _nrow;}
  size_t ncol()const{return _ncol;}
  double& operator()(size_t i, size_t j){
    return *(_buf+i*_ncol+j);
  }
private:
    size_t _nrow;
    size_t _ncol;
    double *_buf;
};

// use ndtype = np.float64 to avoid copy.(be careful)
void printArray(py::array_t<double>ndarray);

void square(py::array_t<double>ndarray);



PYBIND11_MODULE(numpyToCPP, m) {
    m.def("printArray", &printArray, "print array");
    m.def("square", &square, "square");
} 
#endif
