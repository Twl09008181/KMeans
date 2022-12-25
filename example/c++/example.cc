#include <iostream>
#include <iomanip>
#include <loadMnist.hpp>
#include <kmeans.hpp>
#include <chrono>
#include <cassert>
using namespace std;
int main(int argc,char*argv[]){

    using T = double;
    // prepare data
    dataset<T> mnist = load_mnist<T>("./mnist/train-images-idx3-ubyte","./mnist/train-labels-idx1-ubyte"); 
    dataSetPtr<T> mnistPtr(mnist.num, mnist.dim, &mnist.data[0]);


    // kmeans setting
    int threadNum = 8;
    size_t k = 10;
    bool simd = false;
    bool verbose = true;
    int maxIter = 300;
    double tol = 1e-4;
    kmeans<T> kms(k, maxIter, tol, verbose, simd, threadNum);
    kms.fit(mnistPtr);
    auto centers = kms._cluster_centers;
    auto labels = kms.predict(mnistPtr);
    std::cout<<"inertia:"<<kms._inertia<<"\n";


    for(auto center:centers){
      for(int row = 0; row < 28; ++row){
        std::cout<<"\n";
        for(int col = 0; col < 28; ++col){
          if(center[row*28+col] > 128)
            std::cout<<"1";
          else
            std::cout<<"0";
        }
      }
    }
    return 0;
}
