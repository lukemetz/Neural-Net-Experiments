#pragma once

#include "net.hpp"
#include "net_raw.hpp"
#include "net_raw_utils.hpp"

#include "net_gpu_impl.hpp"
template <typename activation, typename error>
inline void gpu_train_batch(FeedForward_Network<activation, error>& network,
    arma::Mat<float> inputs, arma::Mat<float> targets, float learning_rate, int batch_size) {

  network.resize_activation(batch_size);
  Raw_FeedForward_Network<activation, error> raw_net = convert_to_raw(network);
  Raw_FeedForward_Network<activation, error> * d_network = network_to_gpu(raw_net);
  int input_size = network.input_size;
  int hidden_size = network.hidden_size;
  int output_size = network.output_size;

  int batches_in_train = targets.n_rows/batch_size - 1;
  //batches_in_train = 1000;
  for (int i = 0; i < batches_in_train; ++i) {
    arma::Mat<float> input_slice = inputs.rows(i*batch_size, (i+1) * batch_size - 1);

    Raw_Matrix raw_input = to_raw(input_slice);
    Raw_Matrix * d_input = matrix_to_gpu(raw_input);
    int num_trials = input_slice.n_rows;

    calculate_activation(num_trials, input_size, hidden_size, output_size, d_network, d_input);

    arma::Mat<float> targets_slice = targets.rows(i*batch_size, (i+1) * batch_size - 1);

    Raw_Matrix raw_targets = to_raw(targets_slice);
    Raw_Matrix * d_targets = matrix_to_gpu(raw_targets);

    backprop(num_trials, input_size, hidden_size, output_size, d_network, d_targets, learning_rate);
  }

  network_to_cpu(d_network, raw_net);
  update_from_raw(network, raw_net);
}

template <typename activation, typename error>
inline arma::Mat<float> gpu_predict(FeedForward_Network<activation, error>& network,
    arma::Mat<float> inputs) {
  network.resize_activation(inputs.n_rows);
  Raw_FeedForward_Network<activation, error> raw_net = convert_to_raw(network);
  Raw_FeedForward_Network<activation, error> * d_network = network_to_gpu(raw_net);
  Raw_Matrix raw_inputs = to_raw(inputs);
  Raw_Matrix * d_inputs = matrix_to_gpu(raw_inputs);

  int num_trials = inputs.n_rows;
  int input_size = network.input_size;
  int hidden_size = network.hidden_size;
  int output_size = network.output_size;

  calculate_activation(num_trials, input_size, hidden_size, output_size, d_network, d_inputs);

  network_to_cpu(d_network, raw_net);
  update_from_raw(network, raw_net);
  return network.activation_output;
}
