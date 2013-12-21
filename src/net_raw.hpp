#pragma once
#include "functions.hpp"
#include <cassert>
#include <stdio.h>
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
      printf("GPU(%d, %d) \n", row, col);
    #else
      printf("(%d, %d) \n", row, col);
    #endif
    }
    //assert(row < n_rows && col < n_cols);
    return data[row + col * n_rows];
  }
};

template <typename activation = Logistic, typename error = Squared_Error>
struct Raw_FeedForward_Network {
  int input_size;
  int hidden_size;
  int output_size;

  Raw_Matrix weights_inputToHidden;
  Raw_Matrix weights_hiddenToOutput;

  Raw_Matrix last_weights_inputToHidden;
  Raw_Matrix last_weights_hiddenToOutput;

  Raw_Matrix activation_input;
  Raw_Matrix activation_hidden;
  Raw_Matrix activation_output;

  Raw_Matrix output_deltas;
  Raw_Matrix hidden_deltas;
};
