#include "net.hpp"
#include "net_cpu.hpp"

#include <gtest/gtest.h>

std::array<float, 2> xor_func(float x, float y) {
  std::array<float, 2> res =  {{static_cast<float>((static_cast<bool>(x)^static_cast<bool>(y))),
                               static_cast<float>(!(static_cast<bool>(x)^static_cast<bool>(y)))}};
  return res;
};

template <typename model_t>
void check_xor(model_t f) {
  arma::Mat<float> result;

  result = predict(f, {{0,0}});

  EXPECT_GT(result[1], .9);
  EXPECT_LT(result[0], .1);

  result = predict(f, {{1,1}});
  EXPECT_GT(result[1], .9);
  EXPECT_LT(result[0], .1);

  result = predict(f, {{0,1}});
  EXPECT_LT(result[1], .1);
  EXPECT_GT(result[0], .9);

  result = predict(f, {{1,0}});
  EXPECT_LT(result[1], .1);
  EXPECT_GT(result[0], .9);
}

TEST(FeedForward_Network, reasonable_results_for_online_train_xor) {
  FeedForward_Network<> f({2, 10, 2});
  f.resize_activation(1);
  randomize(f);

  const int num_rows = 40000;
  arma::Mat<float> features(num_rows, 2);
  arma::Mat<float> target(num_rows, 2);

  for (int z=0; z < num_rows/4; z++) {
    for (int i=0; i<2; i++) {
      for (int j=0; j<2; j++) {
        int on_index = z*4+i*2+j;
        features(on_index, 0) = i;
        features(on_index, 1) = j;
        target(on_index, 0) = xor_func(i,j)[0];
        target(on_index, 1) = xor_func(i,j)[1];
      }
    }
  }

  float learning_rate = 0.05f;
  for (int i=0; i < 40; i++) {
    train_online(f, features, target, learning_rate);
  }
  check_xor(f);
}

TEST(FeedForward_Network, reasonable_results_batch_train_xor) {
  FeedForward_Network<> f({2, 10, 2});
  randomize(f);

  const int num_rows = 10000;
  arma::Mat<float> features(num_rows, 2);
  arma::Mat<float> target(num_rows, 2);

  for (int z=0; z < num_rows/4; z++) {
    for (int i=0; i<2; i++) {
      for (int j=0; j<2; j++) {
        int on_index = z*4+i*2+j;
        features(on_index, 0) = i;
        features(on_index, 1) = j;
        target(on_index, 0) = xor_func(i,j)[0];
        target(on_index, 1) = xor_func(i,j)[1];
      }
    }
  }

  float learning_rate = 0.4f;
  int batch_size = 1;
  train_batch(f, features, target, batch_size, learning_rate);
  train_batch(f, features, target, batch_size, learning_rate);
  train_batch(f, features, target, batch_size, learning_rate);
  check_xor(f);
}

TEST(FeedForward_Network, reasonable_results_batch_train_xor_4_layer) {
  FeedForward_Network<> f({2, 8, 8, 2});
  randomize(f);

  const int num_rows = 5000;
  arma::Mat<float> features(num_rows, 2);
  arma::Mat<float> target(num_rows, 2);

  for (int z=0; z < num_rows/4; z++) {
    for (int i=0; i<2; i++) {
      for (int j=0; j<2; j++) {
        int on_index = z*4+i*2+j;
        features(on_index, 0) = i;
        features(on_index, 1) = j;
      }
    }
  }

  shuffle(features);
  for (int i=0; i<features.n_rows; ++i) {
    target(i, 0) = xor_func(features(i, 0), features(i, 1))[0];
    target(i, 1) = xor_func(features(i, 0), features(i, 1))[1];
  }


  float learning_rate = 0.3f;
  int batch_size = 1;
  for (int i=0; i < 100; ++i) {
    train_batch(f, features, target, batch_size, learning_rate);
    //auto predict_dat= predict(f, features);
    //std::cout << "Squared error:" << squared_diff(target, predict_dat) << std::endl;
  }

  check_xor(f);
}

TEST(FeedForward_Network, activations_are_correct_shape) {
  FeedForward_Network<> f({2, 10, 2});
  const int num_rows = 11;
  f.resize_activation(num_rows);
  arma::Mat<float> features(num_rows, 2, arma::fill::zeros);
  calculate_activation(f, features);

  ASSERT_EQ(f.activations[0].n_rows, 11);
  ASSERT_EQ(f.activations[1].n_rows, 11);
  ASSERT_EQ(f.activations[2].n_rows, 11);
}
