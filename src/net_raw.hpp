#pragma once

struct Raw_Matrix {
  int n_rows;
  int n_cols;
  float * data;
};

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
};
