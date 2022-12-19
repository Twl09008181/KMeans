#ifndef ALIGNMENT_HPP
#define ALIGNMENT_HPP

#include <stdlib.h>
template <class T>
struct alignedAllocator
{

  // AVX instruction need 32 bytes-aligment
  alignedAllocator(size_t alignment=32)
    :Alignment{alignment}{}
  const size_t Alignment;
  using value_type = T;

  //aligned_alloc(size_ alignment, size_t size)
  //Parameters
  //alignment	-	specifies the alignment. Must be a valid alignment supported by the implementation.
  //size	-	number of bytes to allocate. An integral multiple of alignment
  T* allocate(std::size_t n){
    T *p = static_cast<T*>(
        aligned_alloc(Alignment,
          (n*sizeof(T)/Alignment+1)*Alignment)// size must be multiple of alignment
        );
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


template<typename T>
using alignedVectorType = std::vector<T, alignedAllocator<T>>;

#endif
