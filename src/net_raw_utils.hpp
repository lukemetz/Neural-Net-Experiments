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
  raw.input_size = network.input_size;
  raw.hidden_size = network.hidden_size;
  raw.output_size = network.output_size;

  raw.weights_inputToHidden = to_raw(network.weights_inputToHidden);
  raw.weights_hiddenToOutput = to_raw(network.weights_hiddenToOutput);

  raw.last_weights_inputToHidden = to_raw(network.last_weights_inputToHidden);
  raw.last_weights_hiddenToOutput = to_raw(network.last_weights_hiddenToOutput);

  raw.activation_input = to_raw(network.activation_input);
  raw.activation_hidden = to_raw(network.activation_hidden);
  raw.activation_output = to_raw(network.activation_output);
  return raw;
}

inline arma::Mat<float> from_raw(const Raw_Matrix & raw) {
  return arma::Mat<float>(raw.data, raw.n_rows, raw.n_cols);
}

template <typename activation, typename error>
inline void update_from_raw(FeedForward_Network<activation, error> & network, const Raw_FeedForward_Network<activation, error> &raw) {
  network.input_size = raw.input_size;
  network.hidden_size = raw.hidden_size;
  network.output_size = raw.output_size;

  network.weights_inputToHidden = from_raw(raw.weights_inputToHidden);
  network.weights_hiddenToOutput = from_raw(raw.weights_hiddenToOutput);

  network.last_weights_inputToHidden = from_raw(raw.last_weights_inputToHidden);
  network.last_weights_hiddenToOutput = from_raw(raw.last_weights_hiddenToOutput);

  network.activation_input = from_raw(raw.activation_input);
  network.activation_hidden = from_raw(raw.activation_hidden);
  network.activation_output = from_raw(raw.activation_output);
}
