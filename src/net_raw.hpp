#pragma once
#include "functions.hpp"
#include <cassert>
#include <stdio.h>
#include <vector>

struct Raw_Matrix {
  int n_rows;
  int n_cols;
  float * data;
  #ifdef __NVCC__
  __device__
  #endif
  inline float & at(int row, int col) {
    if (row >= n_rows && col >= n_cols) {
      #ifdef __NVCC__
        printf("ERROR in writing at: GPU(%d, %d) \n", row, col);
      #else
        printf("ERROR in writing at: (%d, %d) \n", row, col);
      #endif
    }
    //assert(row < n_rows && col < n_cols);
    return data[row + col * n_rows];
  }
};

//TODO fix leaks here
template <typename activation = Logistic, typename error = Squared_Error>
struct Raw_FeedForward_Network {
  int * layer_sizes;
  int num_layers;

  Raw_Matrix * weights;
  Raw_Matrix * last_weights;
  Raw_Matrix * activations;
  Raw_Matrix * deltas;
};
