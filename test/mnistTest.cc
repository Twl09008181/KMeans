#include <loadMnist.hpp>
#include <iostream>
#include <iomanip>
int main(int argc,char*argv[]){
    dataset mnist = load_mnist("./mnist/train-images.idx3-ubyte","./mnist/train-labels.idx1-ubyte"); 
    showMnist(mnist);



    return 0;
}
