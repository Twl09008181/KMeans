#!/bin/bash
rm speed.txt
echo "speed test" >> speed.txt
echo "simd 32 100 1"
echo "simd 32 100 1" >> speed.txt
python3 speed.py simd 32 100 1 >> speed.txt 
echo "simd 32 100 2"
echo "simd 32 100 2" >> speed.txt
python3 speed.py simd 32 100 2 >> speed.txt 
echo "simd 32 100 4"
echo "simd 32 100 4" >> speed.txt
python3 speed.py simd 32 100 4 >> speed.txt 
echo "simd 32 100 8"
echo "simd 32 100 8" >> speed.txt
python3 speed.py simd 32 100 8 >> speed.txt 
echo "simd 32 100 16"
echo "simd 32 100 16" >> speed.txt
python3 speed.py simd 32 100 16 >> speed.txt 

echo "simd 64 100 1"
echo "simd 64 100 1" >> speed.txt
python3 speed.py simd 64 100 1 >> speed.txt
echo "simd 64 100 2"
echo "simd 64 100 2" >> speed.txt
python3 speed.py simd 64 100 2 >> speed.txt
echo "simd 64 100 4"
echo "simd 64 100 4" >> speed.txt
python3 speed.py simd 64 100 4 >> speed.txt
echo "simd 64 100 8"
echo "simd 64 100 8" >> speed.txt
python3 speed.py simd 64 100 8 >> speed.txt
echo "simd 64 100 16"
echo "simd 64 100 16" >> speed.txt
python3 speed.py simd 64 100 16 >> speed.txt

echo "sk 32 100 1"
echo "sk 32 100 1" >> speed.txt
OMP_NUM_THREADS=1 python3 speed.py sklearn 32 100 1 >> speed.txt 
echo "sk 32 100 2"
echo "sk 32 100 2" >> speed.txt
OMP_NUM_THREADS=2 python3 speed.py sklearn 32 100 2 >> speed.txt 
echo "sk 32 100 4"
echo "sk 32 100 4" >> speed.txt
OMP_NUM_THREADS=4 python3 speed.py sklearn 32 100 4 >> speed.txt 
echo "sk 32 100 8"
echo "sk 32 100 8" >> speed.txt
OMP_NUM_THREADS=8 python3 speed.py sklearn 32 100 8 >> speed.txt 
echo "sk 32 100 16"
echo "sk 32 100 16" >> speed.txt
OMP_NUM_THREADS=16 python3 speed.py sklearn 32 100 16 >> speed.txt 

echo "sk 64 100 1"
echo "sk 64 100 1" >> speed.txt
OMP_NUM_THREADS=1 python3 speed.py sklearn 64 100 1 >> speed.txt 
echo "sk 64 100 2"
echo "sk 64 100 2" >> speed.txt
OMP_NUM_THREADS=2 python3 speed.py sklearn 64 100 2 >> speed.txt 
echo "sk 64 100 4"
echo "sk 64 100 4" >> speed.txt
OMP_NUM_THREADS=4 python3 speed.py sklearn 64 100 4 >> speed.txt 
echo "sk 64 100 8"
echo "sk 64 100 8" >> speed.txt
OMP_NUM_THREADS=8 python3 speed.py sklearn 64 100 8 >> speed.txt 
echo "sk 64 100 16"
echo "sk 64 100 16" >> speed.txt
OMP_NUM_THREADS=16 python3 speed.py sklearn 64 100 16 >> speed.txt 

echo "nsimd 32 100 1"
echo "nsimd 32 100 1" >> speed.txt
python3 speed.py nosimd 32 100 1 >> speed.txt 
echo "nsimd 32 100 2"
echo "nsimd 32 100 2" >> speed.txt
python3 speed.py nosimd 32 100 2 >> speed.txt 
echo "nsimd 32 100 4"
echo "nsimd 32 100 4" >> speed.txt
python3 speed.py nosimd 32 100 4 >> speed.txt 
echo "nsimd 32 100 8"
echo "nsimd 32 100 8" >> speed.txt
python3 speed.py nosimd 32 100 8 >> speed.txt 
echo "nsimd 32 100 16"
echo "nsimd 32 100 16" >> speed.txt
python3 speed.py nosimd 32 100 16 >> speed.txt 

echo "nsimd 64 100 1"
echo "nsimd 64 100 1" >> speed.txt
python3 speed.py nosimd 64 100 1 >> speed.txt 
echo "nsimd 64 100 2"
echo "nsimd 64 100 2" >> speed.txt
python3 speed.py nosimd 64 100 2 >> speed.txt 
echo "nsimd 64 100 4"
echo "nsimd 64 100 4" >> speed.txt
python3 speed.py nosimd 64 100 4 >> speed.txt 
echo "nsimd 64 100 8"
echo "nsimd 64 100 8" >> speed.txt
python3 speed.py nosimd 64 100 8 >> speed.txt 
echo "nsimd 64 100 16"
echo "nsimd 64 100 16" >> speed.txt
python3 speed.py nosimd 64 100 16 >> speed.txt 

