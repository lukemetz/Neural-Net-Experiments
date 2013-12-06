#include "net.hpp"
#include <armadillo>
#include <iostream>

int main(int argc, char * argv[]) {
  arma::Mat<float> A;
  A.load("mnist.arma");
  A.shed_row(0); //remove header labels
  const int n_rows = 40000;
  const int output_cols = 10;
  const int test_num = 2000;
  arma::Mat<float> labels(n_rows, output_cols, arma::fill::zeros);
  arma::Mat<float> test_labels(test_num, output_cols, arma::fill::zeros);
  int i=0;
  int j=0;
  for (auto iter = A.begin_col(0); iter != A.end_col(0); ++iter) {
    if (i >= n_rows) {
      if (j >= test_num) {
        break;
      }
      test_labels(j, *iter) = 1;
      j += 1;
      continue;
    }
    labels(i, *iter) = 1;
    i += 1;
  }


  std::cout << labels << std::endl;
  A.shed_col(0);
  A /= 256;
  A -= .5;
  arma::Mat<float> test = A.rows(n_rows, n_rows + test_num - 1);

  A.shed_rows(n_rows, A.n_rows-1);
  std::cout << A.n_rows << "," << labels.n_rows << std::endl;
  arma::Mat<float> data = join_rows(labels, A);
  const int feature_size = 784;
  FeedForward_Network<> f(feature_size, 1000, 10);
  randomize(f);
  std::cout << "train" << std::endl;
  for (int i=0; i <80; i++) {
    std::cout << i << std::endl;
    train_batch(f, data.cols(output_cols, data.n_cols - 1).rows(0, n_rows-1), data.cols(0, output_cols-1), 0.03f, 10);
    std::cout << "start score" << std::endl;
    arma::Mat<float> result = predict(f, data.cols(output_cols, data.n_cols-1).rows(0, n_rows-1));
    std::cout << "Score: " << classify_percent_score(result.t(), data.cols(0, output_cols-1)) << std::endl;

    result = predict(f, test);
    std::cout << "TestS: " << classify_percent_score(result.t(), test_labels) << std::endl;
    std::cout << "Shuffling" << std::endl;
    data = shuffle(data);
  }

  return 0;
}
