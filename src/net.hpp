#pragma once
#include "functions.hpp"
#include <armadillo>
#include <algorithm>
#include <random>
#include <cassert>
#include <vector>

template <typename activation = Logistic, typename error = Squared_Error>
struct FeedForward_Network {
  FeedForward_Network(std::vector<int> layer_sizes_in) : layer_sizes(layer_sizes_in) {

    //For the connections inbetween layers
    for (int i = 0; i < layer_sizes.size()-1; ++i) {
      weights.push_back(arma::Mat<float>(layer_sizes[i], layer_sizes[i+1], arma::fill::zeros));
      last_weights.push_back(arma::Mat<float>(layer_sizes[i], layer_sizes[i+1], arma::fill::zeros));
    }
  }

  inline void resize_activation(int num_trials) {
    activations.clear();
    for (int i = 0; i < layer_sizes.size(); ++i) {
      activations.push_back(arma::Mat<float>(num_trials, layer_sizes[i], arma::fill::zeros));
    }

    //No deltas for input layer
    deltas.clear();
    for (int i = 0; i < layer_sizes.size() - 1; ++i) {
      deltas.push_back(arma::Mat<float>(num_trials, layer_sizes[i+1]));
    }
  }

  std::vector<int> layer_sizes;

  std::vector<arma::Mat<float>> weights;
  std::vector<arma::Mat<float>> last_weights;
  std::vector<arma::Mat<float>> activations;
  std::vector<arma::Mat<float>> deltas;
};
