#include "net_gpu_impl.hpp"
#include <host_config.h>
#include <iostream>
#include <stdio.h>

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

Raw_FeedForward_Network * network_to_gpu(Raw_FeedForward_Network & network) {
  Raw_FeedForward_Network h_network = network;
  
  //copy_matrix_with_gpu_ptr(h_network.weights_inputToHidden, network.weights_inputToHidden);
  cudaMalloc((void **) &(h_network.weights_inputToHidden.data),
      matrix_size(h_network.weights_inputToHidden));

  cudaMemcpy(h_network.weights_inputToHidden.data, h_network.weights_inputToHidden.data,
      matrix_size(h_network.weights_inputToHidden), cudaMemcpyHostToDevice);

  copy_matrix_with_gpu_ptr(h_network.weights_hiddenToOutput, network.weights_hiddenToOutput);
  
  copy_matrix_with_gpu_ptr(h_network.activation_input, network.activation_input);
  copy_matrix_with_gpu_ptr(h_network.activation_hidden, network.activation_hidden);
  copy_matrix_with_gpu_ptr(h_network.activation_output, network.activation_output);

  Raw_FeedForward_Network * d_network;
  cudaMalloc((void **)&d_network, sizeof(Raw_FeedForward_Network));
  cudaMemcpy(d_network, &h_network, sizeof(Raw_FeedForward_Network), cudaMemcpyHostToDevice);

  return d_network;
}

void network_to_cpu(Raw_FeedForward_Network * d_network,
    Raw_FeedForward_Network & h_network) {
  Raw_FeedForward_Network orig_address_network = h_network;
  cudaMemcpy(&h_network, d_network, sizeof(Raw_FeedForward_Network), cudaMemcpyDeviceToHost);
  Raw_FeedForward_Network gpu_address = h_network;

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


__global__ void kernel_raw_predict(Raw_FeedForward_Network * d_network, Raw_Matrix * input, Raw_Matrix * output) {
  printf ("Hello from inside kernel %d\n", blockIdx.x);
}

Raw_Matrix raw_predict_gpu(Raw_FeedForward_Network & network, Raw_Matrix & input) {
  /*
  Raw_FeedForward_Network * d_network = to_gpu(network);
  Raw_Matrix * d_output;
  cudaMalloc((void *jjj*) &d_output, sizeof(Raw_Matrix));
  Raw_Matrix * d_input  = malloc_Matrix(input);
  cudaMalloc((void **) &d_output->data, sizeof(float) * network.output_size * input.n_rows);

  kernel_raw_predict<<<input.n_rows, 1>>>(d_network, d_input, d_output);
  */
  Raw_Matrix r;
  return r;
}

void raw_train_batch_gpu(Raw_FeedForward_Network & network, Raw_Matrix & inputs,
    Raw_Matrix & targets, float learning_rate, int batch_size) {

}
