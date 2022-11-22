#!/bin/bash
python3 build/speed.py sklearn 32 10
python3 build/speed.py sklearn 64 10
python3 build/speed.py simd 32 10
python3 build/speed.py simd 64 10
python3 build/speed.py nosimd 32 10
python3 build/speed.py nosimd 64 10
