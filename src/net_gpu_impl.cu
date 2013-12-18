#include "net_gpu_impl.hpp"
#include <host_config.h>
#include <iostream>
#include <stdio.h>
#include "functions.hpp"

inline size_t matrix_size(const Raw_Matrix & m) {
  return sizeof(float) * m.n_rows * m.n_cols;
}

Raw_Matrix * matrix_to_gpu(Raw_Matrix & source) {
  Raw_Matrix h_mat;
  cudaMalloc((void **) &(h_mat.data), matrix_size(source));
  cudaMemcpy(h_mat.data, source.data, matrix_size(source), cudaMemcpyHostToDevice);
  h_mat.n_rows = source.n_rows;
  h_mat.n_cols = source.n_cols;
  Raw_Matrix *d_ma;
  cudaMalloc((void **) &d_ma, sizeof(Raw_Matrix));
  cudaMemcpy(d_ma, &h_mat, sizeof(Raw_Matrix), cudaMemcpyHostToDevice);
  return d_ma;
}

Raw_Matrix matrix_to_cpu(Raw_Matrix * d_mat) {
  Raw_Matrix h_mat;
  cudaMemcpy(&h_mat, d_mat, sizeof(Raw_Matrix), cudaMemcpyDeviceToHost);
  float * d_data = h_mat.data;
  h_mat.data = new float[h_mat.n_rows * h_mat.n_cols];
  cudaMemcpy(h_mat.data, d_data, matrix_size(h_mat), cudaMemcpyDeviceToHost);
  return h_mat;
}

void copy_matrix_with_gpu_ptr(Raw_Matrix & dst, Raw_Matrix & src) {
  cudaMalloc((void **) &(dst.data), matrix_size(src));
  cudaMemcpy(dst.data, src.data, matrix_size(src), cudaMemcpyHostToDevice);
}

template<typename activation, typename error>
Raw_FeedForward_Network<activation, error> * network_to_gpu(Raw_FeedForward_Network<activation, error> & network) {
  Raw_FeedForward_Network<activation, error> h_network = network;

  copy_matrix_with_gpu_ptr(h_network.weights_inputToHidden, network.weights_inputToHidden);

  copy_matrix_with_gpu_ptr(h_network.weights_hiddenToOutput, network.weights_hiddenToOutput);

  copy_matrix_with_gpu_ptr(h_network.activation_input, network.activation_input);
  copy_matrix_with_gpu_ptr(h_network.activation_hidden, network.activation_hidden);
  copy_matrix_with_gpu_ptr(h_network.activation_output, network.activation_output);

  Raw_FeedForward_Network<activation, error> * d_network;
  cudaMalloc((void **)&d_network, sizeof(Raw_FeedForward_Network<activation, error>));
  cudaMemcpy(d_network, &h_network, sizeof(Raw_FeedForward_Network<activation, error>), cudaMemcpyHostToDevice);

  return d_network;
}

template<typename activation, typename error>
void network_to_cpu(Raw_FeedForward_Network<activation, error> * d_network,
    Raw_FeedForward_Network<activation, error> & h_network) {
  Raw_FeedForward_Network<activation, error> orig_address_network = h_network;
  cudaMemcpy(&h_network, d_network, sizeof(Raw_FeedForward_Network<activation, error>), cudaMemcpyDeviceToHost);
  Raw_FeedForward_Network<activation, error> gpu_address = h_network;

  h_network.weights_inputToHidden.data = orig_address_network.weights_inputToHidden.data;
  h_network.weights_hiddenToOutput.data = orig_address_network.weights_hiddenToOutput.data;
  h_network.activation_input.data = orig_address_network.activation_input.data;
  h_network.activation_hidden.data = orig_address_network.activation_hidden.data;
  h_network.activation_output.data = orig_address_network.activation_output.data;

  cudaMemcpy(h_network.weights_inputToHidden.data, gpu_address.weights_inputToHidden.data,
      matrix_size(h_network.weights_inputToHidden), cudaMemcpyDeviceToHost);

  cudaMemcpy(h_network.weights_hiddenToOutput.data, gpu_address.weights_hiddenToOutput.data,
      matrix_size(h_network.weights_hiddenToOutput), cudaMemcpyDeviceToHost);

  cudaMemcpy(h_network.activation_input.data, gpu_address.activation_input.data,
      matrix_size(h_network.activation_input), cudaMemcpyDeviceToHost);

  cudaMemcpy(h_network.activation_hidden.data, gpu_address.activation_hidden.data,
      matrix_size(h_network.activation_hidden), cudaMemcpyDeviceToHost);

  cudaMemcpy(h_network.activation_output.data, gpu_address.activation_output.data,
      matrix_size(h_network.activation_output), cudaMemcpyDeviceToHost);
}


const int block_size = 64;

template<typename activation, typename error>
__global__ void kernel_calculate_hidden_activations(Raw_FeedForward_Network<activation, error> * d_network) {
  
  Raw_Matrix & prior_activation = d_network->activation_input;
  Raw_Matrix & post_activation = d_network->activation_hidden;
  Raw_Matrix & weights = d_network->weights_inputToHidden;
  
  int index = blockIdx.x * block_size + threadIdx.x;
  if (index >= post_activation.n_rows * prior_activation.n_cols) {
    return;
  }
  int on_activation= index / prior_activation.n_rows;
  int on_trial= index % post_activation.n_rows;
  post_activation.at(on_activation, on_trial) = 0;
  for (int i=0; i < weights.n_rows; ++i) {
    post_activation.at(on_trial, on_activation) += weights.at(i, on_activation) * prior_activation.at(on_trial, i);
  }
  post_activation.at(on_trial, on_activation) = activation::activation(post_activation.at(on_trial, on_activation));
}

template<typename activation, typename error>
__global__ void kernel_calculate_output_activations(Raw_FeedForward_Network<activation, error> * d_network) {

  Raw_Matrix & prior_activation = d_network->activation_hidden;
  Raw_Matrix & post_activation = d_network->activation_output;
  Raw_Matrix & weights = d_network->weights_hiddenToOutput;
  int index = blockIdx.x * block_size + threadIdx.x;
  if (index >= post_activation.n_rows * prior_activation.n_cols) {
    return;
  }
  int on_activation= index / prior_activation.n_rows;
  int on_trial= index % post_activation.n_rows;
  post_activation.at(on_activation, on_trial) = 0;
  for (int i=0; i < weights.n_rows; ++i) {
    post_activation.at(on_trial, on_activation) += weights.at(i, on_activation) * prior_activation.at(on_trial, i);
  }
  post_activation.at(on_trial, on_activation) = activation::activation(post_activation.at(on_trial, on_activation));

}

template<typename activation, typename error>
__global__ void kernel_set_input_activations(Raw_FeedForward_Network<activation, error> * d_network, Raw_Matrix * input) {

  int index = blockIdx.x * block_size + threadIdx.x;
  int num_trials = input->n_rows;
  int feature_size = input->n_cols;
  if (index >= feature_size * num_trials) {
    return;
  }
  int on_trial = index / input->n_cols;
  int on_activation = index % input->n_cols;

  d_network->activation_input.at(on_trial, on_activation) = input->at(on_trial, on_activation);
}

template<typename activation, typename error>
void calculate_activation(int num_trials, int input_size, int hidden_size, int output_size, Raw_FeedForward_Network<activation, error> * d_network, Raw_Matrix * d_input)
{
  kernel_set_input_activations<<<1 + num_trials * input_size / block_size, block_size>>>(d_network, d_input);
  kernel_calculate_hidden_activations<<<1 + num_trials * hidden_size / block_size, block_size>>>(d_network);
  kernel_calculate_output_activations<<<1 + num_trials * output_size / block_size, block_size>>>(d_network);
}

template Raw_FeedForward_Network<Logistic, Squared_Error> * network_to_gpu(Raw_FeedForward_Network<Logistic, Squared_Error> & source);
template void network_to_cpu(Raw_FeedForward_Network<Logistic, Squared_Error> * d_network,
    Raw_FeedForward_Network<Logistic, Squared_Error> & h_network);

template void calculate_activation(int num_trials, int input_size, int hidden_size, int output_size, Raw_FeedForward_Network<Logistic, Squared_Error> * d_network, Raw_Matrix * d_input);
