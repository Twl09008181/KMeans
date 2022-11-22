#ifndef LOADMNIST_HPP
#define LOADMNIST_HPP

#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <fstream>



template<typename T>
struct dataset{
  std::vector<T>data;
  std::vector<unsigned>label;
  long num;
  long dim;
};
template<typename T>
dataset<T> load_mnist(std::string ImageFile,std::string LabelFile);
template<typename T>
void showMnist(const dataset<T>&mnist);


int reverseInt (int i);


template<typename T>
void readImage(std::string ImageFile, dataset<T>& out){
    std::ifstream Image (ImageFile,std::ios::binary);
    if (Image.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        Image.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        Image.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        Image.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        Image.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        // save
        out.num = number_of_images;
        out.dim = n_rows * n_cols; // each data has out.dim pixels.
        out.data.resize(out.num * out.dim);
        for(long i=0;i<number_of_images;++i){
          for(long d = 0; d < out.dim; d++){
            unsigned char temp=0;
            Image.read((char*)&temp,sizeof(temp));
            out.data.at(i * out.dim + d) = static_cast<T>(temp);
          }
        }
    }
    else{
        std::cout<<"can't open"<< ImageFile <<"\n";
        exit(1);
    }
    Image.close();
}

template<typename T>
void readLabel(std::string LabelFile, dataset<T>& out){
    std::ifstream Label (LabelFile,std::ios::binary);
    if(Label.is_open()){
        int magic_number=0;
        int number_of_labels=0;
        Label.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        Label.read((char*)&number_of_labels,sizeof(number_of_labels));
        number_of_labels= reverseInt(number_of_labels);
        out.label.resize(number_of_labels);
        for(int i = 0;i<number_of_labels;i++){
            unsigned char l=0;
            Label.read((char*)&l,sizeof(l));
            out.label.at(i) = static_cast<unsigned>(l);
        }
    }else{
        std::cout<<"can't open"<<LabelFile<<"\n";exit(1);
    }
}


template<typename T>
dataset<T> load_mnist(std::string ImageFile,std::string LabelFile){
    dataset<T> mnist;
    readImage<T>(ImageFile, mnist);
    readLabel(LabelFile, mnist);
    return mnist;
}


template<typename T>
void showMnist(const dataset<T>&mnist){
    const auto &X = mnist.data;
    const auto &Y = mnist.label;
    for(size_t i = 0;i < mnist.num;i++){
      std::cout<<"\nlabel:"<<Y.at(i)<<"\n";
      for(size_t d = 0;d < mnist.dim; ++d){
        if(d % 28 == 0)
          std::cout<<"\n";
        if(X[i*mnist.dim+d] > 128){
          std::cout<<"1 ";
        }
        else{
          std::cout<<"0 ";
        }
      } 
    }
}

#endif
