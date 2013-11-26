#include "net.hpp"
#include <iostream>
int main(int argc, char * argv[]) {
  FeedForward f;
  f.randomize();

  Array2D<float, 4, 2> X;
  Array2D<float, 4, 2> Y;

  //A simple XOR
  X(0,0) = 0;
  X(0,1) = 1;
  Y(0,0) = 0;
  Y(0,1) = 1;

  X(1,0) = 1;
  X(1,1) = 0;
  Y(1,0) = 0;
  Y(1,1) = 1;

  X(2,0) = 1;
  X(2,1) = 1;
  Y(2,0) = 0;
  Y(2,1) = 1;

  X(3,0) = 0;
  X(3,1) = 0;
  Y(3,0) = 0;
  Y(3,1) = 0;

  auto func = [](float x, float y) -> std::array<float, 2> {
    //std::array<float, 2> res =  {x*y, x*x+y};
    std::array<float, 2> res =  {float(bool(x)^bool(y)), float(!(bool(x)^bool(y)))};

    return res;
  };

  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      std::array<float, 2> Test = {float(i), float(j)};
      auto result = f.predict(Test);
      std::cout << "{ " << result[0] << ", " << result[1]  << "}" << std::endl;
      std::cout << "Expected {" << func(i, j)[0] << "," << func(i,j)[1]  << "}" << std::endl;
    }
  }
  std::cout << "!~!!!!!~~~" << std::endl;

  for (int z=0; z < 10000; z++) {
    for (int i=0; i<2; i++) {
      for (int j=0; j<2; j++) {
        f.fit({float(i),float(j)}, func(i, j));
      }
    }
  }

  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      std::array<float, 2> Test = {float(i), float(j)};
      auto result = f.predict(Test);
      std::cout << "{ " << result[0] << ", " << result[1]  << "}" << std::endl;
      std::cout << "Expected {" << func(i, j)[0] << "," << func(i, j)[1]  << "}" << std::endl;
    }
  }


  std::cout << "Hello" << std::endl;

  return 0;
}
