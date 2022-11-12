#include "../include/numpyToCPP.hpp"
// use ndtype = np.float64 to avoid copy.(be careful)
void printArray(py::array_t<double>ndarray){
  py::buffer_info info = ndarray.request();
  std::cout<<"ndim:"<<info.ndim<<"\n";
  size_t size = 1;
  for(int dim = 0; dim < info.ndim; dim++){
    std::cout<<"shape["<<dim<<"]="<<info.shape[dim]<<"\n";
    size *= info.shape[dim];
  }
  double *ptr = static_cast<double*>(info.ptr);
  for(size_t i = 0; i < size; i++){
    std::cout<<*(ptr+i)<<" ";
  }
  std::cout<<"\n";
}

void square(py::array_t<double>ndarray){
  py::buffer_info info = ndarray.request();
  size_t size = 1;
  for(int dim = 0; dim < info.ndim; dim++){
    size *= info.shape[dim];
  }
  double *ptr = static_cast<double*>(info.ptr);
  for(size_t i = 0; i < size; i++){
    *(ptr+i) = *(ptr+i) * *(ptr+i);
  }
}
