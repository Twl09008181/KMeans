#include <immintrin.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

template <class T>
struct alignedAllocator
{
  using value_type = T;

  T* allocate(std::size_t n){
    T *p = static_cast<T*>(aligned_alloc(32, (n*sizeof(float)/32+1)*32));
    if(p)
      return p;
    else
      throw std::bad_alloc();
  }

  void deallocate(T* p, std::size_t n){
    free(p);
  }

};


double distanceSqaure(float *v1, float *v2, long N){

  __m256 sumBuffer = _mm256_set1_ps(0);
  size_t step = 8;// 8 floats = 64 bytes.
  size_t i = 0;
  for(;i+step <= N; i+=step){ 
    __m256 p1 = _mm256_load_ps(v1+i);
    __m256 p2 = _mm256_load_ps(v2+i);
    __m256 sub1 = _mm256_sub_ps(p1, p2);
    __m256 sub2 = _mm256_sub_ps(p1, p2);
    sumBuffer = _mm256_add_ps(sumBuffer, _mm256_mul_ps(sub1, sub2));
  }

  alignas(32) float result[step];
  _mm256_store_ps(result, sumBuffer);
  double remain = 0;
  for(;i < N;i++){
    remain += (v1[i]-v2[i])*(v1[i]-v2[i]);
  }
  double s = 0;
  for(int j = 0; j < step; j++)
    s+=result[j];
  return s + remain;
}


int main(int argc, char **argv){
  int N = 784;
 //alignas(32) float a[N];
 //alignas(32) float b[N];
 //float*ptr1 = a;
 //float*ptr2 = b;


  //float = 4 bytes, 32 = 8 floats
  //N*sizeof(float) is real memory we need
  //it's not always multiple of 32
  //(N*sizeof(float) / 32 + 1) * 32
  //float *ptr1 = static_cast<float*>(aligned_alloc(32, (N*sizeof(float)/32+1)*32));
  //float *ptr2 = static_cast<float*>(aligned_alloc(32, (N*sizeof(float)/32+1)*32));
//  for(int i = 0;i < N; i++){
//    ptr1[i] = i+2;
//    ptr2[i] = i;
//  }

  std::vector<float, alignedAllocator<float>>v1(N);
  std::vector<float, alignedAllocator<float>>v2(N);
  for(int i = 0; i < N; i++)
  {
     v1[i] = i+1;
     v2[i] = i;
  }
  std::cout<<distanceSqaure(&v1[0],&v2[0],N);

  return 0;
}
