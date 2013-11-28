#include <iostream>
#include <string>
#include <armadillo>

int main(int argc, char * argv[])
{
  if (argc != 3) {
    std::cout << "Give in a csv to convert followed by output file" << std::endl;
    return 1;
  }
  std::cout << "Converting CSV to ARMA" << std::endl;
  arma::Mat<float> A;
  A.load(argv[1], arma::csv_ascii);
  A.save(argv[2]);
  return 0;
}
