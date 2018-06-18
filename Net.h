#ifndef __NET_H_INCLUDED__
#define __NET_H_INCLUDED__
#include "Neuron.h"
#include "Matrix.h"

typedef std::vector<Neuron> Layer;

class Net{
public:
  Net(const std::vector<unsigned> &TOPOLOGY, const float ALPHA);
  void showConnections();
  void showNeurons(unsigned type);
  std::vector<float> output(unsigned type);
  void forwardPropagation(std::vector<float> &input);
  void backPropagation(std::vector<float> &results);
private:
  Matrix toVector(Layer &l, unsigned type);
  float _alpha;
  std::vector<Layer> _net;
  std::vector<Matrix> _connections;
};

#endif
