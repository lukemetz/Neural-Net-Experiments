#pragma once
#include "functions.hpp"
#include <armadillo>
#include <algorithm>
#include <random>
#include <cassert>

template <typename activation = Logistic, typename error = Squared_Error>
struct FeedForward_Network {
  FeedForward_Network(int input_size_in, int hidden_size_in, int output_size_in) :
    input_size(input_size_in), hidden_size(hidden_size_in), output_size(output_size_in){

    weights_inputToHidden = arma::Mat<float>(input_size, hidden_size, arma::fill::zeros);
    weights_hiddenToOutput = arma::Mat<float>(hidden_size, output_size, arma::fill::zeros);
    last_weights_inputToHidden = arma::Mat<float>(input_size, hidden_size, arma::fill::zeros);
    last_weights_hiddenToOutput = arma::Mat<float>(hidden_size, output_size, arma::fill::zeros);
    activation_input = arma::Mat<float>(1, input_size, arma::fill::zeros);
    activation_hidden = arma::Mat<float>(1, hidden_size, arma::fill::zeros);
    activation_output = arma::Mat<float>(1, output_size, arma::fill::zeros);

    output_deltas = arma::Mat<float>(1, output_size);
    hidden_deltas = arma::Mat<float>(1, hidden_size);
  }

  inline void resize_activation(int num_trials) {
    activation_input = arma::Mat<float>(num_trials, input_size, arma::fill::zeros);
    activation_hidden = arma::Mat<float>(num_trials, hidden_size, arma::fill::zeros);
    activation_output = arma::Mat<float>(num_trials, output_size, arma::fill::zeros);

    output_deltas = arma::Mat<float>(num_trials, output_size);
    hidden_deltas = arma::Mat<float>(num_trials, hidden_size);
  }

  int input_size = 0;
  int hidden_size = 0;
  int output_size = 0;

  arma::Mat<float> weights_inputToHidden;
  arma::Mat<float> weights_hiddenToOutput;

  arma::Mat<float> last_weights_inputToHidden;
  arma::Mat<float> last_weights_hiddenToOutput;

  arma::Mat<float> activation_input;
  arma::Mat<float> activation_hidden;
  arma::Mat<float> activation_output;

  //for backprop
  //TODO move these to a trainer of somekind
  arma::Mat<float> output_deltas;
  arma::Mat<float> hidden_deltas;
};
