#ifndef LOADMNIST_HPP
#define LOADMNIST_HPP

#include <vector>
#include <string>
#include <utility>



struct dataset{
  std::vector<double>data;
  std::vector<unsigned>label;
  long num;
  long dim;
};
dataset load_mnist(std::string ImageFile,std::string LabelFile);
void showMnist(const dataset&mnist);
#endif
