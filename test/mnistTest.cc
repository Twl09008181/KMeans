#include <iostream>
#include <iomanip>
#include <loadMnist.hpp>
#include <kmeans.hpp>
#include <chrono>
#include <cassert>
using namespace std;
int main(int argc,char*argv[]){
    using T = float;
    dataset<T> mnist = load_mnist<T>("./mnist/train-images-idx3-ubyte","./mnist/train-labels-idx1-ubyte"); 
    //showMnist(mnist);
    
    dataSetPtr<T> mnistPtr(mnist.num, mnist.dim, &mnist.data[0]);

    for(int threadNum = 1; threadNum <= 16; ++threadNum){

      size_t k = 10;

      std::cout<<"\nthreadNum: "<< threadNum<<"\n";
      auto start = chrono::steady_clock::now();
      kmeans<T> kms(k,300,1e-4,false, false, threadNum);
      kms.fit(mnistPtr);
      auto centers = kms._cluster_centers;
      auto end = chrono::steady_clock::now();
      std::cout<<"non-simd"<<chrono::duration_cast<chrono::milliseconds>(end - start).count()<<"ms\n";
      std::cout<<"inertia:"<<kms._inertia<<"\n";
      kmeans<T> kmsSimd(k,300,1e-4,false, true, threadNum);
      start = chrono::steady_clock::now();
      kmsSimd.fit(mnistPtr);
      end = chrono::steady_clock::now();
      std::cout<<"simd"<<chrono::duration_cast<chrono::milliseconds>(end - start).count()<<"ms\n";
      std::cout<<"inertia:"<<kmsSimd._inertia<<"\n";
      auto centersSimd = kmsSimd._cluster_centers;


      size_t diff = 0;
      for(size_t c = 0; c < k; ++c){
        for(size_t d = 0; d < mnist.dim; ++ d){
          diff += abs(centers[c][d] - centersSimd[c][d]);
        }
      }
      assert(diff < 1e-4);
      assert(abs(kmsSimd._inertia - kms._inertia)/ kmsSimd._inertia < 1e-4);
    }

    return 0;
}
