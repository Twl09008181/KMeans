
kmeans: src/kmeans.cc include/kmeans.hpp
	mkdir -p build
	g++ -lpthread -fopenmp -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` src/kmeans.cc -o build/kmeans`python3-config --extension-suffix`

build/loadMnist.o: src/loadMnist.cc  include/loadMnist.hpp
	g++  src/loadMnist.cc -I./include/ -c -o build/loadMnist.o

mnistTest: build/loadMnist.o  test/mnistTest.cc
	g++  $^ -I./include/  -o build/mnistTest
	./build/mnistTest

numpyToCPP: src/numpyToCPP.cc include/numpyToCPP.hpp
	mkdir -p build
	g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` src/numpyToCPP.cc -o build/numpyToCPP`python3-config --extension-suffix`

runtest:
	PYTHONPATH=build pytest test/kmeansTest.py


clean:
	rm -r -f build
