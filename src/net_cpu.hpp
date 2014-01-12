#pragma once

#include <armadillo>
#include <algorithm>
#include <random>
#include <cassert>

#include "net.hpp"

template <typename activation = Logistic, typename error = Squared_Error>
void train_online(FeedForward_Network<activation, error>& network,
    arma::Mat<float> inputs, arma::Mat<float> targets, float learning_rate) {
    for (int i = 0; i < targets.n_rows; ++i) {
      calculate_activation(network, inputs.row(i));
      backprop(network, targets.row(i), learning_rate);
    }
}

template <typename activation, typename error>
void train_batch(FeedForward_Network<activation, error>& network,
    arma::Mat<float> inputs, arma::Mat<float> targets, int batch_size, float learning_rate) {
    network.resize_activation(batch_size);

    int batches_in_train = targets.n_rows/batch_size - 1;
    for (int i = 0; i < batches_in_train; ++i) {
      arma::Mat<float> input_slice = inputs.rows(i*batch_size, (i+1) * batch_size-1);
      calculate_activation(network, input_slice);
      arma::Mat<float> target_slice = targets.rows(i*batch_size, (i+1) * batch_size-1);
      backprop(network, target_slice, learning_rate);
    }
}

//Randomize weights in a network.
template <typename activation, typename error>
void randomize(FeedForward_Network<activation, error>& network, float standard_deviation = 0.05) {
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0, standard_deviation);

  auto random_num = [&]() {return distribution(generator);};
  for (int i=0; i < network.weights.size(); ++i) {
    network.weights[i].imbue(random_num);
    //TODO figure out which one is right
    //avoid local nearby local maximum
    network.last_weights[i].imbue(random_num);
    //network.last_weights[i] = network.weights[i];
  }
}

template <typename arma_t, typename activation, typename error>
void backprop(FeedForward_Network<activation, error> &network,
    arma_t target, float learning_rate = 0.8f, float momentum = 0.8f) {
  //Calculate deltas

  //output delta first
  network.deltas.back() = error::error_dir(target, network.activations.back()) % activation::activation_dir(network.activations.back());

  //rest of the delta
  for (int i = network.deltas.size() - 2; i >= 0; --i) {
    network.deltas[i] = (network.deltas[i+1] * network.weights[i+1].t()) % activation::activation_dir(network.activations[i+1]);
  }

  //update weights
  for (int i=0; i < network.weights.size(); ++i) {
    auto & standard_piece = (1 - momentum) * learning_rate * (network.deltas[i].t() * network.activations[i]).t();
    auto & momentum_piece = momentum * (network.weights[i] - network.last_weights[i]);
    arma::Mat<float> delta_weights = standard_piece + momentum_piece;
    network.last_weights[i] = network.weights[i];
    network.weights[i] += delta_weights;
  }
}

template <typename arma_t, typename activation, typename error>
void calculate_activation(FeedForward_Network<activation, error>& network,
    arma_t input) {

  network.activations[0] = input;
  for(int i=1; i < network.activations.size(); ++i) {
    network.activations[i] = network.activations[i-1] * network.weights[i-1];
    network.activations[i] = activation::activation(network.activations[i]);
  }
}

//TODO remove this function??
template <typename activation, typename error>
arma::Mat<float> predict(FeedForward_Network<activation, error>& network,
    arma::Mat<float> input) {
  calculate_activation(network, input);
  return network.activations.back();
}

//Scoring function for classification.
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

//Scoring function that calculates the difference in squares between two matrices.
inline float squared_diff(arma::Mat<float> result, arma::Mat<float> correct) {
  assert(result.n_cols == correct.n_cols);
  auto error_diff = correct - result;
  return arma::accu(error_diff % error_diff);
}
