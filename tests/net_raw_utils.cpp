#include "net.hpp"
#include "net_raw_utils.hpp"

#include <gtest/gtest.h>

TEST(raw_network, FeedForward_to_raw_and_back) {
  FeedForward_Network<> f(10, 20, 3);
  f.weights_inputToHidden[3] = 1;
  f.weights_hiddenToOutput[3] = 2;
  f.activation_input[3] = 3;
  f.activation_input[4] = 4;
  f.activation_input[5] = 5;

  Raw_FeedForward_Network raw = convert_to_raw(f);
  ASSERT_EQ(f.input_size, raw.input_size);
  ASSERT_EQ(f.output_size, raw.output_size);
  ASSERT_EQ(f.hidden_size, raw.hidden_size);

  ASSERT_EQ(raw.weights_inputToHidden.data[3], f.weights_inputToHidden[3]);
  ASSERT_EQ(raw.weights_hiddenToOutput.data[3], f.weights_hiddenToOutput[3]);

  ASSERT_EQ(raw.activation_input.data[3], f.activation_input[3]);
  ASSERT_EQ(raw.activation_hidden.data[3], f.activation_hidden[3]);
  ASSERT_EQ(raw.activation_output.data[3], f.activation_output[3]);


  raw.input_size = 11;
  raw.output_size = 21;
  raw.hidden_size = 4;
  raw.weights_inputToHidden.data[3] = 3;
  raw.weights_hiddenToOutput.data[3] = 4;
  raw.activation_input.data[3] = 3;
  raw.activation_hidden.data[4] = 4;
  raw.activation_output.data[5] = 5;

  update_from_raw(f, raw);

  ASSERT_EQ(f.input_size, 11);
  ASSERT_EQ(f.output_size, 21);
  ASSERT_EQ(f.hidden_size, 4);

  ASSERT_EQ(f.weights_inputToHidden[3], 3);
  ASSERT_EQ(f.weights_hiddenToOutput[3], 4);

  ASSERT_EQ(f.activation_input[3], 3);
  ASSERT_EQ(f.activation_hidden[4], 4);
  ASSERT_EQ(f.activation_output[5], 5);
}
