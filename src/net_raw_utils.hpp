#pragma once

#include <armadillo>
#include "net_raw.hpp"
#include "net.hpp"

inline Raw_Matrix to_raw(arma::Mat<float> & mat) {
  Raw_Matrix matrix;
  matrix.n_rows = mat.n_rows;
  matrix.n_cols= mat.n_cols;
  matrix.data = mat.memptr();
  return matrix;
}

template <typename activation, typename error>
inline Raw_FeedForward_Network<activation, error> convert_to_raw(FeedForward_Network<activation, error> & network) {

  Raw_FeedForward_Network<activation, error> raw;

  raw.num_layers = network.layer_sizes.size();
  raw.layer_sizes = new int [raw.num_layers];
  for (int i=0; i < raw.num_layers; ++i) {
    raw.layer_sizes[i] = network.layer_sizes[i];
  }

  raw.weights = new Raw_Matrix [network.weights.size()];
  raw.last_weights = new Raw_Matrix [network.last_weights.size()];
  for (int i=0; i < network.weights.size(); ++i) {
    raw.weights[i] = to_raw(network.weights[i]);
    raw.last_weights[i] = to_raw(network.last_weights[i]);
  }

  raw.activations = new Raw_Matrix [network.activations.size()];
  for (int i=0; i < network.activations.size(); ++i) {
    raw.activations[i] = to_raw(network.activations[i]);
  }

  raw.deltas = new Raw_Matrix [network.deltas.size()];
  for (int i=0; i < network.deltas.size(); ++i) {
    raw.deltas[i] = to_raw(network.deltas[i]);
  }
  return raw;
}

inline arma::Mat<float> from_raw(const Raw_Matrix & raw) {
  return arma::Mat<float>(raw.data, raw.n_rows, raw.n_cols);
}

template <typename activation, typename error>
inline void update_from_raw(FeedForward_Network<activation, error> & network, const Raw_FeedForward_Network<activation, error> &raw) {
  for (int i=0; i < raw.num_layers; ++i) {
    network.layer_sizes[i] = raw.layer_sizes[i];
  }

  for(int i=0; i < network.weights.size(); ++i) {
    network.weights[i] = from_raw(raw.weights[i]);
    network.last_weights[i] = from_raw(raw.last_weights[i]);
  }

  for(int i=0; i < network.activations.size(); ++i) {
    network.activations[i] = from_raw(raw.activations[i]);
  }

  for(int i=0; i < network.deltas.size(); ++i) {
    network.deltas[i] = from_raw(raw.deltas[i]);
  }
}

