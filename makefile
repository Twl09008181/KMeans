
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

build/cppTest: test/mnistTest.cc build/loadMnist.o   build/kmeans.o
	$(CXX) $(CXXFLAGS)  $^ -I./include/  -o $@

cppTest:build/cppTest
	./build/mnistTest

pyTest:build/kmeans.so
	PYTHONPATH=build pytest test/kmeansTest.py


clean:
	rm -r -f build
