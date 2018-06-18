#include "Matrix.h"

Matrix::Matrix(unsigned cols, unsigned rows, bool initConnections) : _matrix(cols, std::vector<float> (rows, 0)) {
  if(initConnections){ // for all neurons (except bias) set incoming conn. weights to values between -1 and 1
    for(unsigned i = 0; i < _matrix.size(); i++){
      for(unsigned j = 0; j < _matrix[0].size() - 1; j++){
        _matrix[i][j] = std::rand() / float(RAND_MAX) * 2 - 1;
      }
    }
  }
}

Matrix Matrix::transpose(){
  Matrix mT(_matrix[0].size(), _matrix.size(), 0);
  for(unsigned i = 0; i < _matrix.size(); i++){
    for(unsigned j = 0; j < _matrix[0].size(); j++){
      mT._matrix[j][i] = _matrix[i][j];
    }
  }
  return mT;
}

Matrix Matrix::dotProduct(Matrix &m2){
  assert(_matrix.size() == m2._matrix[0].size());
  Matrix product(m2._matrix.size(), _matrix[0].size(), 0);
  for(unsigned k = 0; k < _matrix[0].size(); k++){
    for(unsigned i = 0; i < m2._matrix.size(); i++){
      for(unsigned j = 0; j < _matrix.size(); j++){
        product._matrix[i][k] += _matrix[j][k] * m2._matrix[i][j];
      }
    }
  }
  return product;
}

void Matrix::sum(Matrix &m2){
  assert(_matrix.size() == m2._matrix.size() && _matrix[0].size() == m2._matrix[0].size());
  for(unsigned i = 0; i < _matrix.size(); i++){
    for(unsigned j = 0; j < _matrix[0].size(); j++){
      _matrix[i][j] += m2._matrix[i][j];
    }
  }
}

void Matrix::show(){
  for(unsigned i = 0; i < _matrix.size(); i++){
    for(unsigned j = 0; j < _matrix[0].size(); j++){
      std::cout << std::fixed << i + 1 << j + 1 << " " << _matrix[i][j] << "  ";
    }
    std::cout << std::endl;
  }
}
