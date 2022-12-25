
CXX = g++
CXXFLAGS = -std=c++11 -O3 -lpthread -fopenmp


build/kmeans.o: src/kmeans.cc include/kmeans.hpp include/alignedAllocator.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) $< -I./include -c -o $@ -fPIC -march=native

build/kmeans.so: src/pythonKmeans.cc build/kmeans.o
	mkdir -p build
	$(CXX) $(CXXFLAGS) -shared  -fPIC `python3 -m pybind11 --includes` -I./include $^  -o build/kmeans`python3-config --extension-suffix`  -march=native

build/loadMnist.o: src/loadMnist.cc  include/loadMnist.hpp
	mkdir -p build
	$(CXX) src/loadMnist.cc -I./include/ -c -o $@

build/cppTest: test/cppTest.cc build/loadMnist.o   build/kmeans.o
	$(CXX) $(CXXFLAGS)  $^ -I./include/  -o $@

cppTest:build/cppTest
	./build/cppTest

pyTest:build/kmeans.so
	PYTHONPATH=build pytest test/pyTest.py

speedTest:
	PYTHONPATH=build ./speedScript.sh

pyexample:build/kmeans.so
	PYTHONPATH=build python3 example/python/example.py
	
c++example:example/c++/example.cc build/kmeans.o build/loadMnist.o
	$(CXX) $(CXXFLAGS) $^ -I./include  -o $@ 
	./c++example

	


clean:
	rm -r -f build
