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

template <int input_size, int hidden_size, int output_size,
         typename activation = Logistic, typename error = Squared_Error>
struct FeedForward_Network {
  FeedForward_Network() {
    weights_inputToHidden = arma::Mat<float>(input_size, hidden_size);
    weights_hiddenToOutput = arma::Mat<float>(hidden_size, output_size);
    last_weights_inputToHidden = arma::Mat<float>(input_size, hidden_size);
    last_weights_hiddenToOutput = arma::Mat<float>(hidden_size, output_size);
  }

  arma::Mat<float> weights_inputToHidden;
  arma::Mat<float> weights_hiddenToOutput;

  arma::Mat<float> last_weights_inputToHidden;
  arma::Mat<float> last_weights_hiddenToOutput;

  arma::Mat<float> activation_input;
  arma::Mat<float> activation_hidden;
  arma::Mat<float> activation_output;
};

template <int input_size, int hidden_size, int output_size,
         typename activation, typename error>
void train_online(FeedForward_Network<input_size, hidden_size, output_size, activation, error>& network,
    arma::Mat<float> inputs, arma::Mat<float> targets, float learning_rate) {
    for (int i = 0; i < targets.n_rows; ++i) {
      calculate_activation(network, inputs.row(i));
      backprop(network, targets.row(i), learning_rate);
    }
}

template <int input_size, int hidden_size, int output_size,
         typename activation, typename error>
void train_batch(FeedForward_Network<input_size, hidden_size, output_size, activation, error>& network,
    arma::Mat<float> inputs, arma::Mat<float> targets, float learning_rate, int batch_size) {
    int batches_in_train = targets.n_rows/batch_size - 1;
    for (int i = 0; i < batches_in_train; ++i) {
      calculate_activation(network, inputs.rows(i*batch_size, (i+1) * batch_size));
      backprop(network, targets.rows(i*batch_size, (i+1) * batch_size), learning_rate);
    }
}


template <int input_size, int hidden_size, int output_size,
         typename activation, typename error>
void randomize(FeedForward_Network<input_size, hidden_size, output_size, activation, error>& network) {
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0, .01);

  network.weights_inputToHidden.imbue([&]() {return distribution(generator);});
  network.weights_hiddenToOutput.imbue([&]() {return distribution(generator);});
  //TODO does this make sense?
  network.last_weights_inputToHidden.imbue([&]() {return distribution(generator);});
  network.last_weights_hiddenToOutput.imbue([&]() {return distribution(generator);});
}

template <typename arma_t, int input_size, int hidden_size, int output_size,
         typename activation, typename error>
void backprop(FeedForward_Network<input_size, hidden_size, output_size, activation, error> &network,
    arma_t target, float learning_rate = 0.8f) {
  //Calculate deltas
  arma::Mat<float> output_deltas(output_size, network.activation_output.n_cols);
  output_deltas = error::error_dir(target.t(), network.activation_output) % activation::activation_dir(network.activation_output);

  arma::Mat<float> hidden_deltas (hidden_size , network.activation_output.n_cols);
  hidden_deltas = (network.weights_hiddenToOutput * output_deltas) % activation::activation_dir(network.activation_hidden);

  float momentum = 0.8f;
  arma::Mat<float> delta_weights_hiddenToOutput = (1 - momentum) * learning_rate * network.activation_hidden * output_deltas.t() +
    momentum * (network.weights_hiddenToOutput - network.last_weights_hiddenToOutput);
  network.last_weights_hiddenToOutput = network.weights_hiddenToOutput;
  network.weights_hiddenToOutput += delta_weights_hiddenToOutput;

  arma::Mat<float> delta_weights_inputToHidden = (1 - momentum) * learning_rate * network.activation_input * hidden_deltas.t() +
    momentum * (network.weights_inputToHidden - network.last_weights_inputToHidden);
  network.last_weights_inputToHidden = network.weights_inputToHidden;
  network.weights_inputToHidden += delta_weights_inputToHidden;
}

template <typename arma_t, int input_size, int hidden_size, int output_size,
         typename activation, typename error>
void calculate_activation(FeedForward_Network<input_size, hidden_size, output_size, activation, error>& network,
    arma_t input) {
  network.activation_input = input.t();

  network.activation_hidden = (network.weights_inputToHidden.t() * network.activation_input);
  network.activation_hidden = activation::activation(network.activation_hidden);

  network.activation_output = network.weights_hiddenToOutput.t() * network.activation_hidden;
  network.activation_output = activation::activation(network.activation_output);
}

template <int input_size, int hidden_size, int output_size,
         typename activation, typename error>
arma::Mat<float> predict(FeedForward_Network<input_size, hidden_size, output_size, activation, error>& network,
    arma::Mat<float> input) {
  calculate_activation(network, input);
  return network.activation_output;
}

inline double classify_percent_score(arma::Mat<float> result, arma::Mat<float> correct) {
  assert(result.n_cols == correct.n_cols);
  int num_correct = 0;
  for (int i=0; i < result.n_rows; ++i) {
    auto sort_vec = arma::sort_index(result.row(i), 1);
    if (correct.row(i)[sort_vec[0]] == 1) {
      num_correct += 1;
    }
  }
  return static_cast<float>(num_correct) / static_cast<float>(result.n_rows);
}
