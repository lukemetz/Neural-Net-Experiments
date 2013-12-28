#include "net.hpp"

#include <gtest/gtest.h>


TEST(FeedForward_Network, resize) {
  FeedForward_Network<> f(10, 20, 30);
  f.resize_activation(12);
  ASSERT_EQ(f.activation_input.n_rows, 12);
  ASSERT_EQ(f.activation_hidden.n_rows, 12);
  ASSERT_EQ(f.activation_output.n_rows, 12);

  ASSERT_EQ(f.activation_input.n_cols, 10);
  ASSERT_EQ(f.activation_hidden.n_cols, 20);
  ASSERT_EQ(f.activation_output.n_cols, 30);
}
