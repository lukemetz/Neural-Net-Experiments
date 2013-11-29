#include "net.hpp"
#include <armadillo>
#include <iostream>

int main(int argc, char * argv[]) {
  arma::Mat<float> A;
  A.load("mnist.arma");
  A.shed_row(0); //remove header labels
  const int n_rows = 100;
  arma::Mat<float> labels(n_rows, 10, arma::fill::zeros);
  int i=0;
  for (auto iter = A.begin_col(0); iter != A.end_col(0); ++iter) {
    if (i >= n_rows) {
      break;
    }
    labels(i, *iter) = 1;
    i += 1;
  }
  std::cout << labels << std::endl;
  A.shed_col(0);
  const int feature_size = 784;

  FeedForward_Network<feature_size, 1000, 10> f;
  randomize(f);
  std::cout << "train" << std::endl;
  A /= 256;
  A -= .5;
  for (int i=0; i <100; i++) {
    std::cout << i << std::endl;
    train(f, A, labels, 0.1f);
  }
    std::cout << "predict" << std::endl;
    std::cout << predict(f, A.rows(0,10)).t() << std::endl;
    std::cout << labels << std::endl;

  return 0;
}
