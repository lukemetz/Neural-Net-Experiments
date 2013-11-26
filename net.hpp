#include <armadillo>
#include <algorithm>
#include <random>

template<typename T, int x_size, int y_size>
struct Array2D {
  std::array<T, x_size * y_size> data;
  T& operator() (int x, int y) {
    return data.at(x + x_size*y);
  }
};

struct Squared_Error {
  static inline float error(float target, float result) {
    return 1/2.0f * (target - result) * (target - result);
  }
  static inline float error_dir(float target, float result) {
    return (target - result);
  }
};

struct Logistic {
  static inline float activation(float k) {
    return 1 / (1 + exp(-k));
  }

  static inline float activation_dir(float k) {
    return k * (1 - k);
  }
};

template <int input_size, int hidden_size, int output_size,
         typename activation = Logistic, typename error = Squared_Error>
struct FeedForward_Network{
  float learning_rate = 0.8f;

  Array2D<float, input_size, hidden_size> weights_inputToHidden;
  Array2D<float, hidden_size, output_size> weights_hiddenToOutput;

  std::array<float, input_size> activation_input;
  std::array<float, hidden_size> activation_hidden;
  std::array<float, output_size> activation_output;

  FeedForward_Network(float learning_rate) : learning_rate(learning_rate) {}

  void train(std::array<float, input_size> input, std::array<float, output_size> target) {
      calculate_activation(input);
      backprop(target);
  }

  void randomize() {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0, .1);

    for (int i=0; i < input_size; ++i) {
      for (int j=0; j < hidden_size; ++j) {
        weights_inputToHidden(i,j) = distribution(generator);
      }
    }

    for (int i=0; i < hidden_size; ++i) {
      for (int j=0; j < output_size; ++j) {
        weights_hiddenToOutput(i,j) = distribution(generator);
      }
    }
  }

  void backprop(std::array<float, output_size> target) {
    //Calculate deltas
    std::array<float, output_size> output_deltas;
    for (int i=0; i < output_size; ++i) {
      output_deltas[i] = error::error_dir(target[i], activation_output[i]) * activation::activation_dir(activation_output[i]);
    }
    std::array<float, hidden_size> hidden_deltas;
    for (int i=0; i < hidden_size; ++i) {
      float error_sum= 0;
      for (int k=0; k < output_size; ++k) {
        error_sum += output_deltas[k] * weights_hiddenToOutput(i, k);
      }
      hidden_deltas[i] = error_sum * activation::activation_dir(activation_hidden[i]);
    }

    //Update internal weights
    for (int k=0; k < hidden_size; ++k) {
      for (int i=0; i < output_size; ++i) {
        weights_hiddenToOutput(k,i) += learning_rate * output_deltas[i] * activation_hidden[k];
      }
    }

    for (int k=0; k < input_size; ++k) {
      for (int i=0; i < hidden_size; ++i) {
        weights_inputToHidden(k,i) += learning_rate * hidden_deltas[i] * activation_input[k];
      }
    }
  }

  void calculate_activation(std::array<float, input_size> input) {
    activation_input = input;
    for (int i = 0; i < hidden_size; ++i) {
      float temp = 0;
      for (int j = 0; j < input_size; j++) {
        temp += input[j] * weights_inputToHidden(j, i);
      }
      activation_hidden[i] = activation::activation(temp);
    }
    for (int i = 0; i < output_size; ++i) {
      float temp = 0;
      for (int j = 0; j < hidden_size; j++) {
        temp += activation_hidden[j] * weights_hiddenToOutput(j, i);
      }
      activation_output[i] = activation::activation(temp);
    }
  }

  std::array<float, output_size> predict(std::array<float, input_size> input) {
    calculate_activation(input);
    return activation_output;
  }
};
