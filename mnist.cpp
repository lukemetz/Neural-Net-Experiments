#include "net.hpp"
#include <armadillo>
#include <iostream>

int main(int argc, char * argv[]) {
  arma::Mat<float> A;
  A.load("mnist.arma");
  A.shed_row(0); //remove header labels
  const int n_rows = 1000;
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

  FeedForward_Network<feature_size, n_rows, 10> f(0.8f);
  randomize(f);
  std::cout << "train" << std::endl;
  train(f, A, labels);
  std::cout << "predict" << std::endl;
  std::cout << predict(f, A.row(7))[0] << std::endl;
  std::cout << predict(f, A.row(7))[1] << std::endl;
  std::cout << predict(f, A.row(7))[2] << std::endl;
  std::cout << predict(f, A.row(7))[3] << std::endl;
  std::cout << predict(f, A.row(7))[4] << std::endl;
  std::cout << predict(f, A.row(7))[5] << std::endl;
  std::cout << predict(f, A.row(7))[6] << std::endl;
  std::cout << predict(f, A.row(7))[7] << std::endl;
  std::cout << predict(f, A.row(7))[8] << std::endl;
  std::cout << predict(f, A.row(7))[9] << std::endl;

  std::cout << "predict" << std::endl;
  std::cout << predict(f, A.row(0))[0] << std::endl;
  std::cout << predict(f, A.row(0))[1] << std::endl;
  std::cout << predict(f, A.row(0))[2] << std::endl;
  std::cout << predict(f, A.row(0))[3] << std::endl;
  std::cout << predict(f, A.row(0))[4] << std::endl;
  std::cout << predict(f, A.row(0))[5] << std::endl;
  std::cout << predict(f, A.row(0))[6] << std::endl;
  std::cout << predict(f, A.row(0))[7] << std::endl;
  std::cout << predict(f, A.row(0))[8] << std::endl;
  std::cout << predict(f, A.row(0))[9] << std::endl;

  return 0;
}
