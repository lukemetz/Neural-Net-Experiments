#pragma once

#include <armadillo>
#include <algorithm>
#include <random>
#include <cassert>

struct Squared_Error {
  template<typename U>
  static inline U error(U target, U result) {
    return 1/2.0f * (target - result) * (target - result);
  }
  template<typename U, typename J>
  static inline auto error_dir(U target, J result) -> decltype(target-result) {
    return (target - result);
  }
};

using arma::exp;
struct Logistic {
  template<typename U>
  static inline U activation(U k) {
    return 1 / (1 + exp(-k));
  }

  template<typename U>
  static inline U activation_dir(U k) {
    return k % (1 - k);
  }
};

template <typename activation = Logistic, typename error = Squared_Error>
struct FeedForward_Network {
  FeedForward_Network(int input_size_in, int hidden_size_in, int output_size_in) :
    input_size(input_size_in), hidden_size(hidden_size_in), output_size(output_size_in){

    weights_inputToHidden = arma::Mat<float>(input_size, hidden_size);
    weights_hiddenToOutput = arma::Mat<float>(hidden_size, output_size);
    last_weights_inputToHidden = arma::Mat<float>(input_size, hidden_size);
    last_weights_hiddenToOutput = arma::Mat<float>(hidden_size, output_size);
    activation_input = arma::Mat<float>(input_size, 1);
    activation_hidden = arma::Mat<float>(hidden_size, 1);
    activation_output = arma::Mat<float>(output_size, 1);
  }

  int input_size = 0;
  int hidden_size = 0;
  int output_size = 0;

  arma::Mat<float> weights_inputToHidden;
  arma::Mat<float> weights_hiddenToOutput;

  arma::Mat<float> last_weights_inputToHidden;
  arma::Mat<float> last_weights_hiddenToOutput;

  arma::Mat<float> activation_input;
  arma::Mat<float> activation_hidden;
  arma::Mat<float> activation_output;
};
