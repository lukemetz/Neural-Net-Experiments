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

class FeedForward {
public:
  static const int input_size = 2;
  static const int hidden_size = 5;
  static const int output_size = 2;

  const float learning_rate = .8;
  Array2D<float, input_size, hidden_size> weights_inputToHidden;
  Array2D<float, hidden_size, output_size> weights_hiddenToOutput;

  std::array<float, input_size> activation_input;
  std::array<float, hidden_size> activation_hidden;
  std::array<float, output_size> activation_output;

  /*template <std::size_t train_size>
  void fit(Array2D<float, input_size, train_size> input, Array2D<float, output_size, train_size> target) {
    for (int i=0; i < train_size; ++i) {
      auto result = predict(input[i]);
      backprop(target);
    }
  }; */

  void fit(std::array<float, input_size> input, std::array<float, output_size> target) {
      auto result = predict(input);
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
    std::array<float, output_size> output_deltas;
    for (int i=0; i < output_size; ++i) {
      output_deltas[i] = error_dir_function(target[i], activation_output[i]) * activation_dir_func(activation_output[i]);
    }

    std::array<float, hidden_size> hidden_deltas;
    for (int i=0; i < hidden_size; ++i) {
      float error = 0;
      for (int k=0; k < output_size; ++k) {
        error += output_deltas[k] * weights_hiddenToOutput(i, k); //TODO check order
      }
      hidden_deltas[i] = error * activation_dir_func(activation_hidden[i]);
    }

    //update weights
    for (int k=0; k < hidden_size; ++k) {
      for (int i=0; i < output_size; ++i) {
        //TODO check order
        weights_hiddenToOutput(k,i) += learning_rate * output_deltas[i] * activation_hidden[k];
      }
    }

    for (int k=0; k < input_size; ++k) {
      for (int i=0; i < hidden_size; ++i) {
        weights_inputToHidden(k,i) += learning_rate * hidden_deltas[i] * activation_input[k];
      }
    }
  }

  //TODO abstract this math into a structure containing an execute function
  inline float activation_func(float k) {
    //logistic
    return 1 / (1 + exp(-k));
  }

  inline float activation_dir_func(float k) {
    return k * (1 - k);
  }

  inline float error_function(float target, float result) {
    //Squared error
    return 1/2.0f * (target - result) * (target - result);
  }
  inline float error_dir_function(float target, float result) {
    return (target - result);
  }

  /*float delta_weight(float target, float result) {
    //learning rate *
    //derivative of error with respect to activation *
    //derivative of activation with respect to the net input
    //derivative of the net input with respect to a weight
    float dweight = learning_rate * error_dir_function(target, result) * activation_dir_func(value);
  } */

  std::array<float, 2> predict(std::array<float, input_size> input) {
    activation_input = input;
    for (int i = 0; i < hidden_size; ++i) {
      float temp = 0;
      for (int j = 0; j < input_size; j++) {
        temp += input[j] * weights_inputToHidden(j, i);
      }
      activation_hidden[i] = activation_func(temp);
    }
    for (int i = 0; i < output_size; ++i) {
      float temp = 0;
      for (int j = 0; j < hidden_size; j++) {
        temp += activation_hidden[j] * weights_hiddenToOutput(j, i);
      }
      activation_output[i] = activation_func(temp);
    }

    return activation_output;
  };
};
