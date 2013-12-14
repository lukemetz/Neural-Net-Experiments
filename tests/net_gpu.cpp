#include "net.hpp"
#include "net_gpu.hpp"
#include "net_cpu.hpp"
#include "net_raw_utils.hpp"

#include <gtest/gtest.h>

TEST(gpu_utils, raw_to_gpu_and_back) {
  Raw_Matrix r;
  r.n_rows = 10;
  r.n_cols = 30;
  r.data = new float[100];
  for (int i=0; i < 100; i++) {
    r.data[i] = i;
  }

  Raw_Matrix * d_r = matrix_to_gpu(r);
  Raw_Matrix back_r =  matrix_to_cpu(d_r);

  ASSERT_EQ(back_r.n_cols, r.n_cols);
  ASSERT_EQ(back_r.n_rows, r.n_rows);
  for (int i=0; i < 100; i++) {
    ASSERT_EQ(back_r.data[i], r.data[i]);
  }
  delete r.data;
}

TEST(gpu_utils, raw_net_to_gpu_and_back) {
  FeedForward_Network<> f(5, 10, 4);
  randomize(f);
  for (int i = 0; i < 4; ++i) {
    f.activation_input[i] = i*1;
    f.activation_hidden[i] = i*2;
    f.activation_output[i] = i*3;
  }

  Raw_FeedForward_Network raw_net = convert_to_raw(f);

  Raw_FeedForward_Network * d_net = network_to_gpu(raw_net);

  int input_size = raw_net.input_size;
  int output_size = raw_net.output_size;
  int hidden_size = raw_net.hidden_size;
  float inputToHidden4 = raw_net.weights_inputToHidden.data[4];
  float inputToHidden5 = raw_net.weights_inputToHidden.data[5];

  float hiddenToOutput4 = raw_net.weights_hiddenToOutput.data[4];
  float hiddenToOutput5 = raw_net.weights_hiddenToOutput.data[5];

  float hiddenToOutput_n_rows = raw_net.weights_hiddenToOutput.n_rows;
  float hiddenToOutput_n_cols = raw_net.weights_hiddenToOutput.n_cols;
  float inputToHidden_n_rows = raw_net.weights_inputToHidden.n_rows;
  float inputToHidden_n_cols = raw_net.weights_inputToHidden.n_cols;

  //Fudge the data to ensure its properly reset
  raw_net.input_size = 1;
  raw_net.output_size = 2;
  raw_net.hidden_size = 1;

  raw_net.weights_inputToHidden.data[4] = -1;
  raw_net.weights_inputToHidden.data[5] = -1;

  raw_net.weights_hiddenToOutput.data[4] = -1;
  raw_net.weights_hiddenToOutput.data[5] = -1;

  raw_net.weights_hiddenToOutput.n_rows = -1;
  raw_net.weights_hiddenToOutput.n_cols = -1;

  raw_net.weights_inputToHidden.n_rows = -1;
  raw_net.weights_inputToHidden.n_cols = -1;

  for (int i = 0; i < 4; ++i) {
    raw_net.activation_input.data[i] = -1;
    raw_net.activation_hidden.data[i] = -1;
    raw_net.activation_output.data[i] = -1;
  }

  //re copy back from gpu
  network_to_cpu(d_net, raw_net);

  ASSERT_EQ(raw_net.input_size, input_size);
  ASSERT_EQ(raw_net.output_size, output_size);
  ASSERT_EQ(raw_net.hidden_size, hidden_size);

  ASSERT_EQ(raw_net.weights_inputToHidden.data[4], inputToHidden4);
  ASSERT_EQ(raw_net.weights_inputToHidden.data[5], inputToHidden5);

  ASSERT_EQ(raw_net.weights_hiddenToOutput.data[4], hiddenToOutput4);
  ASSERT_EQ(raw_net.weights_hiddenToOutput.data[5], hiddenToOutput5);

  ASSERT_EQ(raw_net.weights_hiddenToOutput.n_rows, hiddenToOutput_n_rows);
  ASSERT_EQ(raw_net.weights_hiddenToOutput.n_cols, hiddenToOutput_n_cols);

  ASSERT_EQ(raw_net.weights_inputToHidden.n_rows, inputToHidden_n_rows);
  ASSERT_EQ(raw_net.weights_inputToHidden.n_cols, inputToHidden_n_cols);

  for (int i = 0; i < 4; ++i) {
    ASSERT_FLOAT_EQ(raw_net.activation_input.data[i], i*1.f);
    ASSERT_FLOAT_EQ(raw_net.activation_hidden.data[i], i*2.f);
    ASSERT_FLOAT_EQ(raw_net.activation_output.data[i], i*3.f);
  }
}
