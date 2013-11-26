#include "net.hpp"

#include <gtest/gtest.h>
TEST(FeedForward_Network, reasonable_results_for_xor) {
  FeedForward_Network<2, 10, 2> f(0.8f);
  randomize(f);

  auto xor_func = [](float x, float y) {
    std::array<float, 2> res =  {{static_cast<float>((static_cast<bool>(x)^static_cast<bool>(y))),
                                 static_cast<float>(!static_cast<bool>(x)^static_cast<bool>(y))}};
    return res;
  };

  for (int z=0; z < 10000; z++) {
    for (int i=0; i<2; i++) {
      for (int j=0; j<2; j++) {
        train(f, {{float(i),float(j)}}, xor_func(i, j));
      }
    }
  }

  std::array<float, 2> result;

  result = predict(f, {{0,0}});
  EXPECT_GT(.9, result[0]);
  EXPECT_LT(.1, result[1]);

  result = predict(f, {{1,1}});
  EXPECT_GT(.9, result[0]);
  EXPECT_LT(.1, result[1]);

  result = predict(f, {{0,1}});
  EXPECT_LT(.1, result[0]);
  EXPECT_GT(.9, result[1]);

  result = predict(f, {{1,0}});
  EXPECT_LT(.1, result[0]);
  EXPECT_GT(.9, result[1]);
}

