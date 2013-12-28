#include "net_gpu_impl.hpp"
#include <host_config.h>
#include <iostream>
#include <stdio.h>
#include "functions.hpp"

#define gpuErr(ans) { gpuAssert((cudaError_t)(ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", (char *) cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline size_t matrix_size(const Raw_Matrix & m) {
  return sizeof(float) * m.n_rows * m.n_cols;
}

Raw_Matrix * matrix_to_gpu(Raw_Matrix & source) {
  Raw_Matrix h_mat;
  gpuErr(cudaMalloc((void **) &(h_mat.data), matrix_size(source)));
  gpuErr(cudaMemcpy(h_mat.data, source.data, matrix_size(source), cudaMemcpyHostToDevice));
  h_mat.n_rows = source.n_rows;
  h_mat.n_cols = source.n_cols;
  Raw_Matrix *d_ma;
  gpuErr(cudaMalloc((void **) &d_ma, sizeof(Raw_Matrix)));
  gpuErr(cudaMemcpy(d_ma, &h_mat, sizeof(Raw_Matrix), cudaMemcpyHostToDevice));
  return d_ma;
}

Raw_Matrix matrix_to_cpu(Raw_Matrix * d_mat) {
  Raw_Matrix h_mat;
  gpuErr(cudaMemcpy(&h_mat, d_mat, sizeof(Raw_Matrix), cudaMemcpyDeviceToHost));
  float * d_data = h_mat.data;
  h_mat.data = new float[h_mat.n_rows * h_mat.n_cols];
  gpuErr(cudaMemcpy(h_mat.data, d_data, matrix_size(h_mat), cudaMemcpyDeviceToHost));
  return h_mat;
}

void free_gpu_matrix(Raw_Matrix * d_mat) {
  gpuErr(cudaFree(d_mat));
}

void copy_matrix_with_gpu_ptr(Raw_Matrix & dst, Raw_Matrix & src) {
  gpuErr(cudaMalloc((void **) &(dst.data), matrix_size(src)));
  gpuErr(cudaMemcpy(dst.data, src.data, matrix_size(src), cudaMemcpyHostToDevice));
}

template<typename activation, typename error>
Raw_FeedForward_Network<activation, error> * network_to_gpu(Raw_FeedForward_Network<activation, error> & network) {
  Raw_FeedForward_Network<activation, error> h_network = network;

  copy_matrix_with_gpu_ptr(h_network.weights_inputToHidden, network.weights_inputToHidden);
  copy_matrix_with_gpu_ptr(h_network.weights_hiddenToOutput, network.weights_hiddenToOutput);
  
  copy_matrix_with_gpu_ptr(h_network.last_weights_inputToHidden, network.last_weights_inputToHidden);
  copy_matrix_with_gpu_ptr(h_network.last_weights_hiddenToOutput, network.last_weights_hiddenToOutput);

  copy_matrix_with_gpu_ptr(h_network.activation_input, network.activation_input);
  copy_matrix_with_gpu_ptr(h_network.activation_hidden, network.activation_hidden);
  copy_matrix_with_gpu_ptr(h_network.activation_output, network.activation_output);
  
  copy_matrix_with_gpu_ptr(h_network.hidden_deltas, network.hidden_deltas);
  copy_matrix_with_gpu_ptr(h_network.output_deltas, network.output_deltas);

  Raw_FeedForward_Network<activation, error> * d_network;
  gpuErr(cudaMalloc((void **)&d_network, sizeof(Raw_FeedForward_Network<activation, error>)));
  gpuErr(cudaMemcpy(d_network, &h_network, sizeof(Raw_FeedForward_Network<activation, error>), cudaMemcpyHostToDevice));

  return d_network;
}

template<typename activation, typename error>
void network_to_cpu_free(Raw_FeedForward_Network<activation, error> * d_network,
    Raw_FeedForward_Network<activation, error> & h_network) {
  Raw_FeedForward_Network<activation, error> orig_address_network = h_network;
  gpuErr(cudaMemcpy(&h_network, d_network, sizeof(Raw_FeedForward_Network<activation, error>), cudaMemcpyDeviceToHost));
  Raw_FeedForward_Network<activation, error> gpu_address = h_network;

  h_network.weights_inputToHidden.data = orig_address_network.weights_inputToHidden.data;
  h_network.weights_hiddenToOutput.data = orig_address_network.weights_hiddenToOutput.data;
  
  h_network.last_weights_inputToHidden.data = orig_address_network.last_weights_inputToHidden.data;
  h_network.last_weights_hiddenToOutput.data = orig_address_network.last_weights_hiddenToOutput.data;

  h_network.activation_input.data = orig_address_network.activation_input.data;
  h_network.activation_hidden.data = orig_address_network.activation_hidden.data;
  h_network.activation_output.data = orig_address_network.activation_output.data;
  
  h_network.output_deltas.data = orig_address_network.output_deltas.data;
  h_network.hidden_deltas.data = orig_address_network.hidden_deltas.data;

#define copy_free(field) \
  gpuErr(cudaMemcpy(h_network.field.data, gpu_address.field.data, \
      matrix_size(h_network.field), cudaMemcpyDeviceToHost)); \
  gpuErr(cudaFree(gpu_address.field.data));
  
  copy_free(weights_inputToHidden);
  copy_free(weights_hiddenToOutput);

  copy_free(last_weights_inputToHidden);
  copy_free(last_weights_hiddenToOutput);
  
  copy_free(activation_input);
  copy_free(activation_hidden);
  copy_free(activation_output);
  
  copy_free(hidden_deltas);
  copy_free(output_deltas);

  gpuErr(cudaFree(d_network));
}


//const int block_size = 128;
const int block_size = 64;


  //TODO unit test this instead of mass of prints

template<typename activation, typename error>
__global__ void kernel_calculate_hidden_activations(Raw_FeedForward_Network<activation, error> * d_network) {

  Raw_Matrix & prior_activation = d_network->activation_input;
  Raw_Matrix & post_activation = d_network->activation_hidden;
  Raw_Matrix & weights = d_network->weights_inputToHidden;

  int index = blockIdx.x * block_size + threadIdx.x;
  if (index >= post_activation.n_rows * post_activation.n_cols) {
    return;
  }
  int on_activation= index / post_activation.n_rows;
  int on_trial= index % post_activation.n_rows;
  //printf("on_activation %d on_trial %d \n", on_activation, on_trial);
  post_activation.at(on_trial, on_activation) = 0;
  for (int i=0; i < weights.n_rows; ++i) {
    post_activation.at(on_trial, on_activation) += weights.at(i, on_activation) * prior_activation.at(on_trial, i);
  }
  post_activation.at(on_trial, on_activation) = activation::activation(post_activation.at(on_trial, on_activation));
  __syncthreads();
}

template<typename activation, typename error>
__global__ void kernel_calculate_output_activations(Raw_FeedForward_Network<activation, error> * d_network) {

  Raw_Matrix & prior_activation = d_network->activation_hidden;
  Raw_Matrix & post_activation = d_network->activation_output;
  Raw_Matrix & weights = d_network->weights_hiddenToOutput;
  int index = blockIdx.x * block_size + threadIdx.x;
  if (index >= post_activation.n_rows * post_activation.n_cols) {
    return;
  }
  int on_activation= index / post_activation.n_rows;
  int on_trial= index % post_activation.n_rows;
  post_activation.at(on_trial, on_activation) = 0;
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
  gpuErr(cudaPeekAtLastError());
  gpuErr( cudaDeviceSynchronize() );
  kernel_calculate_hidden_activations<<<1 + num_trials * hidden_size / block_size, block_size>>>(d_network);
  gpuErr(cudaPeekAtLastError());
  gpuErr( cudaDeviceSynchronize() );
  kernel_calculate_output_activations<<<1 + num_trials * output_size / block_size, block_size>>>(d_network);
  gpuErr(cudaPeekAtLastError());
  gpuErr( cudaDeviceSynchronize() );
}

template<typename activation, typename error>
__global__ void kernel_calculate_output_deltas(Raw_FeedForward_Network<activation, error> * d_network, Raw_Matrix * target) {
  int index = blockIdx.x * block_size + threadIdx.x;
  int num_trials = target->n_rows;
  int feature_size = target->n_cols;
  if (index >= feature_size * num_trials) {
    return;
  }
  int on_trial = index / target->n_cols;
  int on_activation = index % target->n_cols;

  d_network->output_deltas.at(on_trial, on_activation) =
    error::error_dir(target->at(on_trial, on_activation), d_network->activation_output.at(on_trial, on_activation)) *
    activation::activation_dir(d_network->activation_output.at(on_trial, on_activation));
}

template<typename activation, typename error>
__global__ void kernel_calculate_hidden_deltas(Raw_FeedForward_Network<activation, error> * d_network) {
  int index = blockIdx.x * block_size + threadIdx.x;
  int num_trials = d_network->hidden_deltas.n_rows;
  int feature_size = d_network->hidden_deltas.n_cols;
  if (index >= feature_size * num_trials) {
    return;
  }
  int on_trial = index / d_network->hidden_deltas.n_cols;
  int on_activation = index % d_network->hidden_deltas.n_cols;
  int old_activation_length = d_network->output_deltas.n_cols;

  float delta_accumulate = 0;
  for (int i=0; i < old_activation_length; ++i) {
    delta_accumulate +=  d_network->output_deltas.at(on_trial, i) * d_network->weights_hiddenToOutput.at(on_activation, i);
  }

  d_network->hidden_deltas.at(on_trial, on_activation) =
    delta_accumulate * activation::activation_dir(d_network->activation_hidden.at(on_trial, on_activation));
  //printf("output_delta (%d, %d) = %f \n", on_trial, on_activation, d_network->hidden_deltas.at(on_trial, on_activation));
}

template<typename activation, typename error>
__global__ void kernel_update_weights_hiddenToOutput(Raw_FeedForward_Network<activation, error> * d_network, float learning_rate) {
  int index = blockIdx.x * block_size + threadIdx.x;
  int num_prior = d_network->activation_hidden.n_cols;
  int num_post = d_network->activation_output.n_cols;
  int num_trials = d_network->activation_hidden.n_rows;
  Raw_Matrix & weights = d_network->weights_hiddenToOutput;
  Raw_Matrix & last_weights = d_network->last_weights_hiddenToOutput;
  Raw_Matrix & delta = d_network->output_deltas;
  Raw_Matrix & activation_vals = d_network->activation_hidden;

  if (index >= num_post * num_prior) {
    return;
  }

  int on_post= index / num_prior;
  int on_prior= index % num_prior;
  
  float accumulate = 0;
  for (int i=0; i < num_trials; ++i) {
    accumulate += delta.at(i, on_post) * activation_vals.at(i, on_prior);
  }

  float momentum = 0.8f;
  float standard_piece = (1 - momentum) * accumulate * learning_rate;
  float momentum_piece = momentum * (weights.at(on_prior, on_post) - last_weights.at(on_prior, on_post));
  float delta_weight = standard_piece + momentum_piece;
  //printf("accumulate(%d, %d) = %f \n", on_prior, on_post, momentum_piece);
  last_weights.at(on_prior, on_post) = weights.at(on_prior, on_post);

  weights.at(on_prior, on_post) += delta_weight;
}

template<typename activation, typename error>
__global__ void kernel_update_weights_inputToHidden(Raw_FeedForward_Network<activation, error> * d_network, float learning_rate) {
  int index = blockIdx.x * block_size + threadIdx.x;
  int num_trials = d_network->activation_hidden.n_rows;

  Raw_Matrix & weights = d_network->weights_inputToHidden;
  Raw_Matrix & last_weights = d_network->last_weights_inputToHidden;
  Raw_Matrix & delta = d_network->hidden_deltas;
  Raw_Matrix & activation_vals = d_network->activation_input;
  
  int num_prior = activation_vals.n_cols;
  int num_post = delta.n_cols;

  if (index >= num_post * num_prior) {
    return;
  }

  int on_post= index / num_prior;
  int on_prior= index % num_prior;
  //printf("(%d, %d)\n", on_prior, on_post);
  float accumulate = 0;
  for (int i=0; i < num_trials; ++i) {
    accumulate += delta.at(i, on_post) * activation_vals.at(i, on_prior);

  }

  float momentum = 0.8f;
  float standard_piece = (1 - momentum) * accumulate * learning_rate;
  float momentum_piece = momentum * (weights.at(on_prior, on_post) - last_weights.at(on_prior, on_post));
  float delta_weight = standard_piece + momentum_piece;
  last_weights.at(on_prior, on_post) = weights.at(on_prior, on_post);

  weights.at(on_prior, on_post) += delta_weight;
}

template<typename activation, typename error>
void backprop(int num_trials, int input_size, int hidden_size, int output_size,
    Raw_FeedForward_Network<activation, error> * d_network, Raw_Matrix * d_targets, float learning_rate)
{
  kernel_calculate_output_deltas<<<1 + num_trials * output_size / block_size, block_size>>> (d_network, d_targets);
  gpuErr(cudaPeekAtLastError());
  gpuErr( cudaDeviceSynchronize() );

  kernel_calculate_hidden_deltas<<<1 + num_trials * hidden_size/ block_size, block_size>>> (d_network);
  gpuErr(cudaPeekAtLastError());
  gpuErr( cudaDeviceSynchronize() );


  kernel_update_weights_hiddenToOutput
    <<<1 + output_size * hidden_size / block_size, block_size>>>(d_network, learning_rate);
  gpuErr(cudaPeekAtLastError());
  gpuErr( cudaDeviceSynchronize() );

  kernel_update_weights_inputToHidden
    <<<1 + hidden_size * input_size / block_size, block_size>>>(d_network, learning_rate);
  gpuErr(cudaPeekAtLastError());
  gpuErr( cudaDeviceSynchronize() );

}

template Raw_FeedForward_Network<Logistic, Squared_Error> * network_to_gpu(Raw_FeedForward_Network<Logistic, Squared_Error> & source);
template void network_to_cpu_free(Raw_FeedForward_Network<Logistic, Squared_Error> * d_network,
    Raw_FeedForward_Network<Logistic, Squared_Error> & h_network);

template void calculate_activation(int num_trials, int input_size, int hidden_size, int output_size, Raw_FeedForward_Network<Logistic, Squared_Error> * d_network, Raw_Matrix * d_input);

template void backprop(int num_trials, int input_size, int hidden_size, int output_size,
    Raw_FeedForward_Network<Logistic, Squared_Error> * d_network, Raw_Matrix * d_targets, float learning_rate);
