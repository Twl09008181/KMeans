#include <kmeans.hpp>
#include <numeric>
#include <limits>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <vector>
#include <math.h>


void addVec(double* p1, double *p2, long dim){ 
  for(long b = 0; b < dim; ++b){ 
    p1[b]+=p2[b];
  }
}
void scaling(double* p1, double scale, long dim){ 
  for(long b = 0; b < dim; ++b){ 
    p1[b] /= scale;
  }
}

//simd future?
double squareDistance(double* p1, double *p2, long s, long e){ 
  double dis = 0;
  for(long b = s; b < e; ++b){ 
    dis += std::pow(p1[b]-p2[b], 2);
  }
  return dis;
}

double* raw(std::vector<double>&vec){
  return &vec[0];
}





double kmeans(dataSetPtr& data, int k, double epsilon, int maxIter, bool verbose){ 


  srand(0);
  long numOfData = data._dataNum;
  long dim = data._dataDim;
  if(verbose){
    std::cout<<"num of data:"<<numOfData<<"\n";
    std::cout<<"num of dim:"<<dim<<"\n";
  }
  //init
  std::vector<std::vector<double>>clusters(k, std::vector<double>(dim));
  for(long c = 0; c < k; ++c){
    int i = rand() % numOfData;

    if(verbose){
      std::cout<<"\n Initialization cluster "<<c<<" : \n";
    }
    for(long d = 0; d < dim; d++){
      clusters[c][d] = data[i][d];
      if(verbose){
        std::cout<<clusters[c][d]<<" ";
      }
    }
  }
  double difference = std::numeric_limits<double>::max();
  int iters = 0;
  std::vector<std::vector<double>>squareDistances(numOfData, std::vector<double>(k,0));
  double inertia = - 1;
  while(difference > epsilon && iters<maxIter){


    for(long i = 0; i < numOfData; i++){ 
      for(int c = 0; c < k;++c){ 
          squareDistances[i][c] = squareDistance(data[i], raw(clusters[c]), 0, dim);
      }
    }

    std::vector<std::vector<double>>nextClusters(k, std::vector<double>(dim,0));
    std::vector<long>clusterSize(k,0);
    inertia = 0;
    //assign data to closet cluster.
    for(long i = 0; i < numOfData; i++){
      long closet=0;
      double minimumDist = std::numeric_limits<double>::max();
      for(long c = 0; c < k; c++){
        if(squareDistances[i][c] < minimumDist){
          minimumDist = squareDistances[i][c];
          closet = c;
        }
      }
      addVec(raw(nextClusters[closet]), data[i], dim);
      clusterSize[closet]++;
      inertia += minimumDist;
    }



    if(verbose){
      std::cout<<"\nIteration "<<iters<<", inertia "<<inertia<<".\n";
    }

    //update clusters
    for(long c = 0; c < k; c++){
      scaling(raw(nextClusters[c]), clusterSize[c], dim);
    }

    difference = 0;
    for(long c = 0; c < k; c++){
      difference += squareDistance(raw(nextClusters[c]), raw(clusters[c]), 0,dim);
    }
    difference = sqrt(difference);
    
    clusters = std::move(nextClusters);
    iters++;
  }

  if(verbose){
    std::cout<<"Iterations:"<<iters<<"\n";
    for(long c = 0; c < k; ++c){
      std::cout<<"cluster"<<c<<":\n";
      for(long d = 0; d < dim; d++)
        std::cout<<clusters[c][d]<<" ";
      std::cout<<"\n";
    }
    std::cout<<"inertia:"<<inertia<<"\n";

  }
  return inertia;


}



