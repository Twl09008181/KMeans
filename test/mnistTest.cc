#include <iostream>
#include <iomanip>
#include <loadMnist.hpp>
#include <kmeans.hpp>
int main(int argc,char*argv[]){
    dataset mnist = load_mnist("./mnist/train-images.idx3-ubyte","./mnist/train-labels.idx1-ubyte"); 
    //showMnist(mnist);
    
    dataSetPtr mnistPtr(mnist.num, mnist.dim, &mnist.data[0]);
    kmeans(mnistPtr,10,1e-4,300,true); 


    return 0;
}
