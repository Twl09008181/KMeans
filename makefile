
kmeans: src/kmeans.cc include/kmeans.hpp
	mkdir -p build
	g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` src/kmeans.cc -o build/kmeans`python3-config --extension-suffix`



numpyToCPP: src/numpyToCPP.cc include/numpyToCPP.hpp
	mkdir -p build
	g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` src/numpyToCPP.cc -o build/numpyToCPP`python3-config --extension-suffix`

runTest:test/
	PYTHONPATH=build pytest test/test_nptcc.py


clean:
	rm -r -f build