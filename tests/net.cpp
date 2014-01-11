#include "net.hpp"

#include <gtest/gtest.h>


TEST(FeedForward_Network, resize) {
  FeedForward_Network<> f({10, 20, 30});
  f.resize_activation(12);
  ASSERT_EQ(f.activations[0].n_rows, 12);
  ASSERT_EQ(f.activations[1].n_rows, 12);
  ASSERT_EQ(f.activations[2].n_rows, 12);

  ASSERT_EQ(f.activations[0].n_cols, 10);
  ASSERT_EQ(f.activations[1].n_cols, 20);
  ASSERT_EQ(f.activations[2].n_cols, 30);
}
