#pragma once

#include "net.hpp"
#include "net_raw.hpp"
#include "net_raw_utils.hpp"

#include "net_gpu_impl.hpp"

template <typename activation, typename error>
inline void train_batch_gpu(FeedForward_Network<activation, error>& network,
    arma::Mat<float> inputs, arma::Mat<float> targets, float learning_rate, int batch_size) {
  Raw_FeedForward_Network<activation, error> raw_net = convert_to_raw(network);
  auto raw_inputs = to_raw(inputs);
  auto raw_targets = to_raw(targets);
  raw_train_batch_gpu(raw_net, raw_inputs, raw_targets, learning_rate, batch_size);

  update_from_raw(network, raw_net);
}

template <typename activation, typename error>
inline arma::Mat<float> predict_gpu(FeedForward_Network<activation, error>& network,
    arma::Mat<float> inputs) {
  Raw_FeedForward_Network<activation, error> raw_net = convert_to_raw(network);
  auto raw_inputs = to_raw(inputs);
  Raw_Matrix result = raw_predict_gpu(raw_net, raw_inputs);
  return from_raw(result);
}
