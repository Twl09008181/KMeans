#ifndef ALIGNMENT_HPP
#define ALIGNMENT_HPP

#include <stdlib.h>
template <class T>
struct alignedAllocator
{
  using value_type = T;

  T* allocate(std::size_t n){
    T *p = static_cast<T*>(aligned_alloc(32, (n*sizeof(double)/32+1)*32));
    if(p)
      return p;
    else
      throw std::bad_alloc();
  }

  void deallocate(T* p, std::size_t n){
    free(p);
  }
};

template<class T, class U>
bool operator==(const alignedAllocator<T>&a, const alignedAllocator<U>&b){

  return true;
}

template<class T, class U>
bool operator!=(const alignedAllocator<T>&a, const alignedAllocator<U>&b){

  return false;
}

using alignedVector = std::vector<double, alignedAllocator<double>>;

#endif
