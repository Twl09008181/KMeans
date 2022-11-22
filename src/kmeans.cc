#include <kmeans.hpp>
#include <numeric>
#include <limits>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <vector>
#include <math.h>
#include <immintrin.h>
#include <alignedAllocator.hpp>
#include <iterator>


void addVecSIMD(double* v1, double *v2, long N){ 
  size_t step = 4;// 4 double = 64 bytes.
  size_t i = 0;
  for(;i+step <= N; i+=step){ 
    __m256d p1 = _mm256_load_pd(v1+i);
    __m256d p2 = _mm256_load_pd(v2+i);
    __m256d addBuffer = _mm256_add_pd(p1, p2);
    _mm256_store_pd(v1+i, addBuffer);
  }
  for(;i < N;i++){
    v1[i] += v2[i];
  }
}

void addVecSIMD(float* v1, float *v2, long N){ 
  size_t step = 8;// 4 floats = 64 bytes.
  size_t i = 0;
  for(;i+step <= N; i+=step){ 
    __m256 p1 = _mm256_load_ps(v1+i);
    __m256 p2 = _mm256_load_ps(v2+i);
    __m256 addBuffer = _mm256_add_ps(p1, p2);
    _mm256_store_ps(v1+i, addBuffer);
  }
  for(;i < N;i++){
    v1[i] += v2[i];
  }
}
void addVecScalar(double* p1, double *p2, long dim){ 
  for(long b = 0; b < dim; ++b){ 
    p1[b]+=p2[b];
  }
}
void addVecScalar(float* p1, float *p2, long dim){ 
  for(long b = 0; b < dim; ++b){ 
    p1[b]+=p2[b];
  }
}


void addVec(double* v1, double *v2, long N, bool simd){ 
  bool isAlignment = ((intptr_t(v1) & 0x1f)==0) && ((intptr_t(v2) & 0x1f)==0);
  if(isAlignment && simd){
     addVecSIMD(v1, v2, N);
  }
  else{
     addVecScalar(v1, v2, N);
  }
}

void addVec(float* v1, float *v2, long N, bool simd){ 
  bool isAlignment = ((intptr_t(v1) & 0x1f)==0) && ((intptr_t(v2) & 0x1f)==0);
  if(isAlignment && simd){
     addVecSIMD(v1, v2, N);
  }
  else{
     addVecScalar(v1, v2, N);
  }
}

//scalar version 
double squareDistanceScalar(double* p1, double *p2, long s, long e){ 
  double dis = 0;
  for(long b = s; b < e; ++b){ 
    dis += (p1[b]-p2[b]) *(p1[b]-p2[b]);
  }
  return dis;
}

double squareDistanceScalar(float* p1, float *p2, long s, long e){ 
  double dis = 0;
  for(long b = s; b < e; ++b){ 
    dis += (p1[b]-p2[b]) *(p1[b]-p2[b]);
  }
  return dis;
}

//simd
double squareDistanceSIMD(double *v1, double *v2, long s, long e){

  //mutiple accumulators
  __m256d sumBuffer1 = _mm256_set1_pd(0);
  __m256d sumBuffer2 = _mm256_set1_pd(0);
  __m256d sumBuffer3 = _mm256_set1_pd(0);
  __m256d sumBuffer4 = _mm256_set1_pd(0);
  size_t step = 4;// 4 double = 64 bytes.
  size_t i = s;
  for(;i+4*step < e; i+=4*step){ 
    __m256d p1 = _mm256_load_pd(v1+i);
    __m256d p2 = _mm256_load_pd(v2+i);
    __m256d sub1 = _mm256_sub_pd(p1, p2);
    __m256d p3 = _mm256_load_pd(v1+i+step);
    __m256d p4 = _mm256_load_pd(v2+i+step);
    __m256d sub2 = _mm256_sub_pd(p3, p4);
    __m256d p5 = _mm256_load_pd(v1+i+2*step);
    __m256d p6 = _mm256_load_pd(v2+i+2*step);
    __m256d sub3 = _mm256_sub_pd(p5, p6);
    __m256d p7 = _mm256_load_pd(v1+i+3*step);
    __m256d p8 = _mm256_load_pd(v2+i+3*step);
    __m256d sub4 = _mm256_sub_pd(p7, p8);
    sumBuffer1 = _mm256_fmadd_pd(sub1, sub1, sumBuffer1);
    sumBuffer2 = _mm256_fmadd_pd(sub2, sub2, sumBuffer2);
    sumBuffer3 = _mm256_fmadd_pd(sub3, sub3, sumBuffer3);
    sumBuffer4 = _mm256_fmadd_pd(sub4, sub4, sumBuffer4);
  }
  sumBuffer1 = _mm256_add_pd(sumBuffer1, sumBuffer2);
  sumBuffer2 = _mm256_add_pd(sumBuffer3, sumBuffer4);
  sumBuffer1 = _mm256_add_pd(sumBuffer1, sumBuffer2);
  double result[step];
  _mm256_store_pd(result, sumBuffer1);
  double sum = 0;
  for(int j = 0; j < step; j++)
    sum+=result[j];
  double remain = 0;
  for(;i < e;i++){
    remain += (v1[i]-v2[i])*(v1[i]-v2[i]);
  }
  return sum+ remain;
}


//simd
double squareDistanceSIMD(float *v1, float *v2, long s, long e){

  //mutiple accumulators
  __m256 sumBuffer1 = _mm256_set1_ps(0);
  __m256 sumBuffer2 = _mm256_set1_ps(0);
  size_t step = 8;// 8 floats= 64 bytes.
  size_t i = s;
  for(;i+2*step < e; i+=2*step){ 
    __m256 p1 = _mm256_load_ps(v1+i);
    __m256 p2 = _mm256_load_ps(v2+i);
    __m256 sub1 = _mm256_sub_ps(p1, p2);
    __m256 p3 = _mm256_load_ps(v1+i+step);
    __m256 p4 = _mm256_load_ps(v2+i+step);
    __m256 sub2 = _mm256_sub_ps(p3, p4);
    sumBuffer1 = _mm256_fmadd_ps(sub1, sub1, sumBuffer1);
    sumBuffer2 = _mm256_fmadd_ps(sub2, sub2, sumBuffer2);
  }
  sumBuffer1 = _mm256_add_ps(sumBuffer1, sumBuffer2);
  float result[step];
  _mm256_store_ps(result, sumBuffer1);
  double sum = 0;
  for(int j = 0; j < step; j++)
    sum+=result[j];
  double remain = 0;
  for(;i < e;i++){
    remain += (v1[i]-v2[i])*(v1[i]-v2[i]);
  }
  return sum+ remain;
}



double squareDistance(double *v1, double *v2, long start, long end, bool simd){
  bool isAlignment = ((intptr_t(v1) & 0x1f)==0) && ((intptr_t(v2) & 0x1f)==0);
  if(isAlignment && simd){
    return squareDistanceSIMD(v1, v2, start, end);
  }
  else{
    return squareDistanceScalar(v1, v2, start, end);
  }
}


double squareDistance(float *v1, float *v2, long start, long end, bool simd){
  bool isAlignment = ((intptr_t(v1) & 0x1f)==0) && ((intptr_t(v2) & 0x1f)==0);
  if(isAlignment && simd){
    return squareDistanceSIMD(v1, v2, start, end);
  }
  else{
    return squareDistanceScalar(v1, v2, start, end);
  }
}


