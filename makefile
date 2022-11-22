build/kmeans.o: src/kmeans.cc include/kmeans.hpp
	mkdir -p build
	g++ -O3  -lpthread -fopenmp -std=c++11 src/kmeans.cc -I./include -c -o build/kmeans.o -fPIC -march=native

build/kmeans.so: build/kmeans.o
	mkdir -p build
	g++ -lpthread -fopenmp -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -I./include src/pythonKmeans.cc build/kmeans.o  -o build/kmeans`python3-config --extension-suffix`  -march=native

build/loadMnist.o: src/loadMnist.cc  include/loadMnist.hpp
	mkdir -p build
	g++  src/loadMnist.cc -I./include/ -c -o build/loadMnist.o

mnistTest: test/mnistTest.cc build/loadMnist.o   build/kmeans.o
	g++  $^ -I./include/  -o build/mnistTest -lpthread -fopenmp -O3 -march=native
	./build/mnistTest

runtest:build/kmeans.so
	PYTHONPATH=build pytest test/kmeansTest.py

cacheTest: build/loadMnist.o cacheTest.cc
	g++ cacheTest.cc -I./include -march=native build/loadMnist.o -o cacheTest

clean:
	rm -r -f build
