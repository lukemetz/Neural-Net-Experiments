#include "net.hpp"
#include <armadillo>
#include <iostream>

int main(int artc, char * argv[]) {
  arma::Mat<float> A;
  A.load("mnist.arma");
  A.shed_row(0); //remove header labels
  const int n_rows = 1000;
  Array2D<float, n_rows, 10> labels;
  int i=0;
  for (auto iter = A.begin_col(0); iter != A.end_col(0); ++iter) {
    if (i >= n_rows) {
      break;
    }
    for (int j = 0; j < 10; ++j) {
      labels(j, *iter) = 0;
    }
    labels(i, *iter) = 1;
    i += 1;
  }
  A.shed_col(0);
  return 0;
}
