#include "net.hpp"

#include <gtest/gtest.h>

std::array<float, 2> xor_func(float x, float y) {
  std::array<float, 2> res =  {{static_cast<float>((static_cast<bool>(x)^static_cast<bool>(y))),
                               static_cast<float>(!(static_cast<bool>(x)^static_cast<bool>(y)))}};
  return res;
};

template <typename model_t>
void check_xor(model_t f) {
  std::array<float, 2> result;

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

TEST(FeedForward_Network, reasonable_results_for_array_xor) {
  FeedForward_Network<2, 10, 2> f(0.8f);
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

  train(f, features, target);
  check_xor(f);
}

