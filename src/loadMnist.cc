#include <iostream>
#include <fstream>
#include <loadMnist.hpp>

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


void readImage(std::string ImageFile, dataset& out){
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
            out.data.at(i * out.dim + d) = static_cast<double>(temp);
          }
        }
    }
    else{
        std::cout<<"can't open"<< ImageFile <<"\n";
        exit(1);
    }
    Image.close();
}

void readLabel(std::string LabelFile, dataset& out){
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


dataset load_mnist(std::string ImageFile,std::string LabelFile){
    dataset mnist;
    readImage(ImageFile, mnist);
    readLabel(LabelFile, mnist);
    return mnist;
}


void showMnist(const dataset&mnist){
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
