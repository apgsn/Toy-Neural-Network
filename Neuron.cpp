#include "Neuron.h"

float Neuron::sigmoid(){
  return 1.0 / (1.0 + std::exp(-_sum));
}

float Neuron::sigmoidDerivative(){
  return sigmoid() * (1.0 - sigmoid());
}

float Neuron::typeSwitcher(unsigned type){
  switch(type){
    case 0: return _id;
    case 1: return _sum;
    case 2: return _output;
    case 3: return _error;
    default: return 0;
  }
}
