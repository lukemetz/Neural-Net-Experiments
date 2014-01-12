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

Raw_Matrix * matrix_array_to_gpu(Raw_Matrix * h_input_array, int size) {
  Raw_Matrix * h_mats = new Raw_Matrix[size];
  for (int i=0; i < size; ++i) {
    h_mats[i].n_rows = h_input_array[i].n_rows;
    h_mats[i].n_cols = h_input_array[i].n_cols;
    copy_matrix_with_gpu_ptr(h_mats[i], h_input_array[i]);
  }
  Raw_Matrix * d_mats;
  gpuErr(cudaMalloc((void **) &(d_mats), sizeof(Raw_Matrix) * size));
  gpuErr(cudaMemcpy(d_mats, h_mats, sizeof(Raw_Matrix) * size, cudaMemcpyHostToDevice));
  delete h_mats;
  return d_mats;
}

template<typename activation, typename error>
Raw_FeedForward_Network<activation, error> * network_to_gpu(Raw_FeedForward_Network<activation, error> & network) {

  Raw_FeedForward_Network<activation, error> h_network;
  h_network.num_layers = network.num_layers;

  h_network.weights      = matrix_array_to_gpu(network.weights,      network.num_layers-1);
  h_network.last_weights = matrix_array_to_gpu(network.last_weights, network.num_layers-1);
  h_network.deltas       = matrix_array_to_gpu(network.deltas,       network.num_layers-1);
  h_network.activations  = matrix_array_to_gpu(network.activations,  network.num_layers);

  int * d_layer_sizes;
  cudaMalloc((void **) &(d_layer_sizes), sizeof(int) * network.num_layers);
  cudaMemcpy(d_layer_sizes, network.layer_sizes, sizeof(int) * network.num_layers , cudaMemcpyHostToDevice);
  h_network.layer_sizes = d_layer_sizes;

  Raw_FeedForward_Network<activation, error> * d_network;
  gpuErr(cudaMalloc((void **)&d_network, sizeof(Raw_FeedForward_Network<activation, error>)));
  gpuErr(cudaMemcpy(d_network, &h_network, sizeof(Raw_FeedForward_Network<activation, error>), cudaMemcpyHostToDevice));

  return d_network;
}

void copy_free(Raw_Matrix * dest, Raw_Matrix * src, int size) {
  //host with device pointers
  Raw_Matrix * hd_matrix_array = new Raw_Matrix[size];
  gpuErr(cudaMemcpy(hd_matrix_array, src, sizeof(Raw_Matrix) * size, cudaMemcpyDeviceToHost));
  for (int i=0; i < size; ++i) {
    gpuErr(cudaMemcpy(dest[i].data, hd_matrix_array[i].data,
        matrix_size(dest[i]), cudaMemcpyDeviceToHost));
    gpuErr(cudaFree(hd_matrix_array[i].data));
  }
  delete hd_matrix_array;
  gpuErr(cudaFree(src));
}

template<typename activation, typename error>
void network_to_cpu_free(Raw_FeedForward_Network<activation, error> * d_network,
    Raw_FeedForward_Network<activation, error> & h_network) {
  Raw_FeedForward_Network<activation, error> hd_copy_back_net;
  gpuErr(cudaMemcpy(&hd_copy_back_net, d_network, sizeof(Raw_FeedForward_Network<activation, error>), cudaMemcpyDeviceToHost));

  copy_free(h_network.weights,      hd_copy_back_net.weights,      h_network.num_layers - 1);
  copy_free(h_network.last_weights, hd_copy_back_net.last_weights, h_network.num_layers - 1);
  copy_free(h_network.deltas,       hd_copy_back_net.deltas,       h_network.num_layers - 1);
  copy_free(h_network.activations,  hd_copy_back_net.activations,  h_network.num_layers    );

  gpuErr(cudaMemcpy(h_network.layer_sizes, hd_copy_back_net.layer_sizes,
        sizeof(int) * h_network.num_layers, cudaMemcpyDeviceToHost));
  gpuErr(cudaFree(d_network));
}
const int block_size = 128;
//const int block_size = 256;
//const int block_size = 512;
//const int block_size = 64;

template<typename activation, typename error>
__global__ void kernel_calculate_layer_activations(Raw_FeedForward_Network<activation, error> * d_network, int on_layer) {

  Raw_Matrix & prior_activation = d_network->activations[on_layer];
  Raw_Matrix & post_activation = d_network->activations[on_layer+1];
  Raw_Matrix & weights = d_network->weights[on_layer];

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
  __syncthreads();
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

  d_network->activations[0].at(on_trial, on_activation) = input->at(on_trial, on_activation);
}

template<typename activation, typename error>
void calculate_activation(int num_trials, std::vector<int> sizes, Raw_FeedForward_Network<activation, error> * d_network, Raw_Matrix * d_input)
{
  int input_size = sizes[0];
  kernel_set_input_activations<<<1 + num_trials * input_size / block_size, block_size>>>(d_network, d_input);
  gpuErr(cudaPeekAtLastError());
  gpuErr( cudaDeviceSynchronize() );
  for (int i = 0; i < sizes.size() - 1; ++i) {
    kernel_calculate_layer_activations<<<1 + num_trials * sizes[i+1] / block_size, block_size>>>(d_network, i);
    gpuErr(cudaPeekAtLastError());
    gpuErr( cudaDeviceSynchronize() );
  }
}

template<typename activation, typename error>
__global__ void kernel_calculate_output_deltas(Raw_FeedForward_Network<activation, error> * d_network, Raw_Matrix * target, int on_layer) {
  int index = blockIdx.x * block_size + threadIdx.x;
  int num_trials = target->n_rows;
  int feature_size = target->n_cols;
  if (index >= feature_size * num_trials) {
    return;
  }
  int on_trial = index / target->n_cols;
  int on_activation = index % target->n_cols;

  d_network->deltas[on_layer].at(on_trial, on_activation) =
    error::error_dir(target->at(on_trial, on_activation), d_network->activations[on_layer+1].at(on_trial, on_activation)) *
    activation::activation_dir(d_network->activations[on_layer+1].at(on_trial, on_activation));
}

template<typename activation, typename error>
__global__ void kernel_calculate_layer_deltas(Raw_FeedForward_Network<activation, error> * d_network, int on_layer) {
  int index = blockIdx.x * block_size + threadIdx.x;
  int num_trials = d_network->deltas[on_layer].n_rows;
  int feature_size = d_network->deltas[on_layer].n_cols;
  if (index >= feature_size * num_trials) {
    return;
  }
  int on_trial = index / d_network->deltas[on_layer].n_cols;
  int on_activation = index % d_network->deltas[on_layer].n_cols;
  int old_activation_length = d_network->deltas[on_layer+1].n_cols;

  float delta_accumulate = 0;
  for (int i=0; i < old_activation_length; ++i) {
    delta_accumulate +=  d_network->deltas[on_layer+1].at(on_trial, i) * d_network->weights[on_layer+1].at(on_activation, i);
  }

  d_network->deltas[on_layer].at(on_trial, on_activation) =
    delta_accumulate * activation::activation_dir(d_network->activations[on_layer+1].at(on_trial, on_activation));
}

template<typename activation, typename error>
__global__ void kernel_update_weights_last(Raw_FeedForward_Network<activation, error> * d_network, float learning_rate, float momentum, int on_layer) {
  int index = blockIdx.x * block_size + threadIdx.x;
  int num_prior = d_network->activations[on_layer].n_cols;
  int num_post = d_network->activations[on_layer+1].n_cols;
  int num_trials = d_network->activations[on_layer].n_rows;
  Raw_Matrix & weights = d_network->weights[on_layer];
  Raw_Matrix & last_weights = d_network->last_weights[on_layer];
  Raw_Matrix & delta = d_network->deltas[on_layer];
  Raw_Matrix & activation_vals = d_network->activations[on_layer];

  if (index >= num_post * num_prior) {
    return;
  }

  int on_post= index / num_prior;
  int on_prior= index % num_prior;
  
  float accumulate = 0;
  for (int i=0; i < num_trials; ++i) {
    accumulate += delta.at(i, on_post) * activation_vals.at(i, on_prior);
  }

  float standard_piece = (1 - momentum) * accumulate * learning_rate;
  float momentum_piece = momentum * (weights.at(on_prior, on_post) - last_weights.at(on_prior, on_post));
  float delta_weight = standard_piece + momentum_piece;
  last_weights.at(on_prior, on_post) = weights.at(on_prior, on_post);

  weights.at(on_prior, on_post) += delta_weight;
}


template<typename activation, typename error>
__global__ void kernel_update_weights_layer(Raw_FeedForward_Network<activation, error> * d_network,
    float learning_rate, float momentum, int on_layer) {
  int index = blockIdx.x * block_size + threadIdx.x;
  int num_trials = d_network->activations[on_layer].n_rows;

  Raw_Matrix & weights = d_network->weights[on_layer];
  Raw_Matrix & last_weights = d_network->last_weights[on_layer];
  Raw_Matrix & delta = d_network->deltas[on_layer];
  Raw_Matrix & activation_vals = d_network->activations[on_layer];
  
  int num_prior = activation_vals.n_cols;
  int num_post = delta.n_cols;

  if (index >= num_post * num_prior) {
    return;
  }

  int on_post= index / num_prior;
  int on_prior= index % num_prior;
  float accumulate = 0;
  for (int i=0; i < num_trials; ++i) {
    accumulate += delta.at(i, on_post) * activation_vals.at(i, on_prior);

  }

  float standard_piece = (1 - momentum) * accumulate * learning_rate;
  float momentum_piece = momentum * (weights.at(on_prior, on_post) - last_weights.at(on_prior, on_post));
  float delta_weight = standard_piece + momentum_piece;
  last_weights.at(on_prior, on_post) = weights.at(on_prior, on_post);

  weights.at(on_prior, on_post) += delta_weight;
}

template<typename activation, typename error>
void backprop(int num_trials, std::vector<int> sizes,
    Raw_FeedForward_Network<activation, error> * d_network, Raw_Matrix * d_targets, float learning_rate, float momentum)
{
  kernel_calculate_output_deltas<<<1 + num_trials * sizes.back() / block_size, block_size>>> (d_network, d_targets, sizes.size()-2);
  gpuErr(cudaPeekAtLastError());
  gpuErr( cudaDeviceSynchronize() );

  for (int i=sizes.size()-3; i >= 0; --i) {
    kernel_calculate_layer_deltas<<<1 + num_trials * sizes[i+1]/ block_size, block_size>>> (d_network, i);
    gpuErr(cudaPeekAtLastError());
    gpuErr( cudaDeviceSynchronize() );
  }


  kernel_update_weights_last
    <<<1 + sizes.back() * sizes[sizes.size() - 2] / block_size, block_size>>>(d_network, learning_rate, momentum, sizes.size()-2);
  gpuErr(cudaPeekAtLastError());
  gpuErr( cudaDeviceSynchronize() );

  for (int i=sizes.size()-3; i >= 0; --i) {
    kernel_update_weights_layer
      <<<1 + sizes[i + 1] * sizes[i]/ block_size, block_size>>>(d_network, learning_rate, momentum, i);
    gpuErr(cudaPeekAtLastError());
    gpuErr( cudaDeviceSynchronize() );
  }
}

//Needed due to template implementations not in header file
#define SPECIALIZE(error, activation) \
template Raw_FeedForward_Network<activation, error> * network_to_gpu(Raw_FeedForward_Network<activation, error> & source); \
template void network_to_cpu_free(Raw_FeedForward_Network<activation, error> * d_network, \
    Raw_FeedForward_Network<activation, error> & h_network); \
template void calculate_activation(int num_trials, std::vector<int> layer_sizes, Raw_FeedForward_Network<activation, error> * d_network, Raw_Matrix * d_input); \
template void backprop(int num_trials, std::vector<int> layer_sizes, \
    Raw_FeedForward_Network<activation, error> * d_network, Raw_Matrix * d_targets, float learning_rate, float momentum);

SPECIALIZE(Squared_Error, Logistic)
SPECIALIZE(Squared_Error, Linear)

#undef SPECIALIZE
