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

    for(int threadNum = 8; threadNum <= 8; ++threadNum){
      std::cout<<"threadNum: "<< threadNum<<"\n";
      auto start = chrono::steady_clock::now();
      kmeans<T> kms(10,300,1e-4,false, false, threadNum);
      kms.fit(mnistPtr);
      auto end = chrono::steady_clock::now();
      std::cout<<"non-simd"<<chrono::duration_cast<chrono::milliseconds>(end - start).count()<<"ms\n";
      std::cout<<"inertia:"<<kms._inertia<<"\n";
      kmeans<T> kmsSimd(10,300,1e-4,false, true, threadNum);
      start = chrono::steady_clock::now();
      kmsSimd.fit(mnistPtr);
      end = chrono::steady_clock::now();
      std::cout<<"simd"<<chrono::duration_cast<chrono::milliseconds>(end - start).count()<<"ms\n";
      std::cout<<"inertia:"<<kmsSimd._inertia<<"\n";

      assert(abs(kmsSimd._inertia - kms._inertia)/ kmsSimd._inertia < 1e-4);
    }

    return 0;
}
