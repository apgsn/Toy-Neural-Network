#include "Net.h"

Net::Net(const std::vector<unsigned> &TOPOLOGY, const float ALPHA){
  _alpha = ALPHA;
  for(unsigned i = 0; i < TOPOLOGY.size(); i++){
    _net.push_back(Layer());
    for(unsigned j = 0; j <= TOPOLOGY[i]; j++){
      _net[i].push_back(Neuron(j));
    }
    if(i) _connections.push_back(Matrix(_net[i - 1].size(), _net[i].size(), 1)); // generate conn. matrix
  }
}

void Net::showConnections(){
  for(Matrix &c : _connections){
    c.show();
  }
}

void Net::showNeurons(unsigned type){
  for(Layer &l : _net){
    for(Neuron &n : l){
      std::cout << std::fixed << n.typeSwitcher(type) << "  ";
    }
    std::cout << std::endl;
  }
}

std::vector<float> Net::output(unsigned type){
  std::vector<float> outputLayer;
  for(Neuron &n : _net.back()){
    outputLayer.push_back(n.typeSwitcher(type));
  }
  outputLayer.pop_back();
  return outputLayer;
}

Matrix Net::toVector(Layer &l, unsigned type){
  Matrix m(1, l.size(), 0);
  for(Neuron &n : l){
    m.value(0, n.id(), n.typeSwitcher(type));
  }
  return m;
}

void Net::forwardPropagation(std::vector<float> &input) {
  assert(input.size() == _net[0].size() - 1);
  for(unsigned layer = 0; layer < _net.size(); layer++) {
    for(Neuron &n : _net[layer]) {
      if(n.id() == _net[layer].size() - 1) continue; // skip bias
      if(!layer){  // input layer
        n.output(input[n.id()]);
      } else {  // other layers
        n.sum(0.0);
        for(Neuron &nPrev : _net[layer - 1]) {
          n.sum(n.sum() + nPrev.output() * _connections[layer - 1].value(nPrev.id(), n.id()));
        }
        n.output(n.sigmoid());
      }
    }
  }
}

void Net::backPropagation(std::vector<float> &results) {
  assert(results.size() == _net.back().size() - 1);
  // step 1: backpropagate error
  for(Neuron &n : _net.back()) { // output layer
    n.error(results[n.id()] - n.output());
  }
  for(unsigned layer = _net.size() - 1; layer > 0; layer--) { // other layers
    Matrix layerAsMatrix = toVector(_net[layer], 3);
    Matrix errHidden = _connections[layer - 1].transpose().dotProduct(layerAsMatrix);
    for(Neuron &n : _net[layer - 1]){
      n.error(errHidden.value(0, n.id()));
    }
    // step 2: update connections
    layerAsMatrix = toVector(_net[layer - 1], 2).transpose();
    Matrix gradients(1, _net[layer].size(), 0);
    for(Neuron &n : _net[layer]){
      if(n.id() == _net[layer].size() - 1) continue; // skip bias
      float gradient = _alpha * n.error() * n.sigmoidDerivative(); // delta rule
      gradients.value(0, n.id(), gradient);
    }
    Matrix delta = gradients.dotProduct(layerAsMatrix);
    _connections[layer - 1].sum(delta);
  }
}
