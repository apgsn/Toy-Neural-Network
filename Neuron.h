#ifndef __NEURON_H_INCLUDED__
#define __NEURON_H_INCLUDED__
#include <cmath>

class Neuron{
public:
  Neuron(unsigned id) : _id(id), _output(1.0) {}
  unsigned id() const { return _id; }
  void output(float output) { _output = output; }
  float output() const { return _output; }
  void error(float error) { _error = error; }
  float error() const { return _error; }
  void sum(float sum) { _sum = sum; }
  float sum() const { return _sum; }
  float sigmoid();
  float sigmoidDerivative();
  float typeSwitcher(unsigned type);
private:
  unsigned _id;
  float _sum, _output, _error;
};

#endif
