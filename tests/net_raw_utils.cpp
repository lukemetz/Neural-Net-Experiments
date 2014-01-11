#include "net.hpp"
#include "net_raw_utils.hpp"

#include <gtest/gtest.h>

TEST(raw_network, FeedForward_to_raw_and_back) {
  FeedForward_Network<> f({10, 20, 3});
  f.resize_activation(10);
  f.weights[0][3] = 1;
  f.weights[1][3] = 2;
  f.activations[0][3] = 3;
  f.activations[0][4] = 4;
  f.activations[0][5] = 5;
  f.deltas[1][3] = 2;
  f.deltas[0][3] = 2;

  Raw_FeedForward_Network<> raw = convert_to_raw(f);
  ASSERT_EQ(f.layer_sizes[0], raw.layer_sizes[0]);
  ASSERT_EQ(f.layer_sizes[2], raw.layer_sizes[2]);
  ASSERT_EQ(f.layer_sizes[1], raw.layer_sizes[1]);

  ASSERT_EQ(raw.weights[0].data[3], f.weights[0][3]);
  ASSERT_EQ(raw.weights[1].data[3], f.weights[1][3]);

  ASSERT_EQ(raw.activations[0].data[3], f.activations[0][3]);
  ASSERT_EQ(raw.activations[1].data[3], f.activations[1][3]);
  ASSERT_EQ(raw.activations[2].data[3], f.activations[2][3]);

  ASSERT_EQ(raw.deltas[1].data[3], f.deltas[1][3]);
  ASSERT_EQ(raw.deltas[0].data[3], f.deltas[0][3]);

  ASSERT_EQ(raw.num_layers, 3);


  raw.layer_sizes[0] = 11;
  raw.layer_sizes[1]= 21;
  raw.layer_sizes[2] = 4;
  raw.weights[0].data[3] = 3;
  raw.weights[1].data[3] = 4;

  raw.weights[0].data[199] = 3;
  raw.weights[1].data[50] = 4;

  raw.activations[0].data[3] = 3;
  raw.activations[1].data[4] = 4;
  raw.activations[2].data[5] = 5;


  raw.deltas[1].data[3] = 5;
  raw.deltas[0].data[3] = 5;

  update_from_raw(f, raw);

  ASSERT_EQ(f.layer_sizes[0], 11);
  ASSERT_EQ(f.layer_sizes[1], 21);
  ASSERT_EQ(f.layer_sizes[2], 4);

  ASSERT_EQ(f.weights[0][3], 3);
  ASSERT_EQ(f.weights[1][3], 4);

  ASSERT_EQ(f.weights[0][199], 3);
  ASSERT_EQ(f.weights[1][50], 4);

  ASSERT_EQ(f.activations[0][3], 3);
  ASSERT_EQ(f.activations[1][4], 4);
  ASSERT_EQ(f.activations[2][5], 5);
  ASSERT_EQ(f.deltas[1][3], 5);
  ASSERT_EQ(f.deltas[0][3], 5);
}

TEST(raw_matrix, Raw_Matrix_set_element) {
  arma::Mat<float> matrix(3, 4);
  for (int i=0; i < 3*4; i++) {
    matrix[i] = i;
  }
  Raw_Matrix raw_mat = to_raw(matrix);
  ASSERT_EQ(0, raw_mat.data[0]);
  ASSERT_EQ(1, raw_mat.data[1]);
  ASSERT_EQ(2, raw_mat.data[2]);

  ASSERT_EQ(matrix.at(0,1), raw_mat.at(0, 1));
  ASSERT_EQ(matrix.at(1,1), raw_mat.at(1, 1));

  raw_mat.at(1, 3) = 100;
  arma::Mat<float> re_mat = from_raw(raw_mat);
  ASSERT_EQ(100, re_mat.at(1, 3));

}
