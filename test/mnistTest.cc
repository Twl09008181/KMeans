#include <iostream>
#include <iomanip>
#include <loadMnist.hpp>
#include <kmeans.hpp>
#include <chrono>
#include <cassert>
using namespace std;
int main(int argc,char*argv[]){
    dataset mnist = load_mnist("./mnist/train-images-idx3-ubyte","./mnist/train-labels-idx1-ubyte"); 
    //showMnist(mnist);
    
    auto start = chrono::steady_clock::now();
    dataSetPtr mnistPtr(mnist.num, mnist.dim, &mnist.data[0]);
    kmeans kms(10,300,1e-4,false, false);
    kms.fit(mnistPtr);
    auto end = chrono::steady_clock::now();
    std::cout<<"non-simd"<<chrono::duration_cast<chrono::milliseconds>(end - start).count()<<"ms\n";
    std::cout<<"inertia:"<<kms._inertia<<"\n";

    kmeans kmsSimd(10,300,1e-4,false, true);
    start = chrono::steady_clock::now();
    kmsSimd.fit(mnistPtr);
    end = chrono::steady_clock::now();
    std::cout<<"simd"<<chrono::duration_cast<chrono::milliseconds>(end - start).count()<<"ms\n";
    std::cout<<"inertia:"<<kmsSimd._inertia<<"\n";

    assert(abs(kmsSimd._inertia - kms._inertia)/ kmsSimd._inertia < 1e-10);
    return 0;
}
