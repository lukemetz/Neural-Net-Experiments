#pragma once

#include "net_raw.hpp"


Raw_Matrix raw_predict_gpu(Raw_FeedForward_Network & network, Raw_Matrix & input);

void raw_train_batch_gpu(Raw_FeedForward_Network & network, Raw_Matrix & inputs,
    Raw_Matrix & targets, float learning_rate, int batch_size);


Raw_Matrix * matrix_to_gpu(Raw_Matrix & source);
Raw_Matrix matrix_to_cpu(Raw_Matrix * d_matrix);

Raw_FeedForward_Network * network_to_gpu(Raw_FeedForward_Network & source);
void network_to_cpu(Raw_FeedForward_Network * d_network, Raw_FeedForward_Network & h_network);

