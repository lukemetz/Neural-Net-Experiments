#pragma once

#include "net_raw.hpp"

template<typename activation, typename error>
Raw_Matrix raw_predict_gpu(Raw_FeedForward_Network<activation, error> & network, Raw_Matrix & input);

template<typename activation, typename error>
void raw_train_batch_gpu(Raw_FeedForward_Network<activation, error> & network,
    Raw_Matrix & inputs, Raw_Matrix & targets, float learning_rate, int batch_size);


Raw_Matrix * matrix_to_gpu(Raw_Matrix & source);
Raw_Matrix matrix_to_cpu(Raw_Matrix * d_matrix);

template<typename activation, typename error>
Raw_FeedForward_Network<activation, error> * network_to_gpu(Raw_FeedForward_Network<activation, error> & source);

template<typename activation, typename error>
void network_to_cpu(Raw_FeedForward_Network<activation, error> * d_network,
    Raw_FeedForward_Network<activation, error> & h_network);

template<typename activation, typename error>
void calculate_activation(int num_trials, int input_size, int hidden_size, int output_size,
    Raw_FeedForward_Network<activation, error> * d_network, Raw_Matrix * d_input);

template<typename activation, typename error>
void backprop(int num_trials, int input_size, int hidden_size, int output_size,
    Raw_FeedForward_Network<activation, error> * d_network, Raw_Matrix * d_targets, float learning_rate = 0.9);
