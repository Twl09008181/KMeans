#!/bin/bash
python3 build/speed.py simd 32
python3 build/speed.py simd 64
python3 build/speed.py nosimd 32
python3 build/speed.py nosimd 64
python3 build/speed.py sklearn 32
python3 build/speed.py sklearn 64
