#ifndef __MATRIX_H_INCLUDED__
#define __MATRIX_H_INCLUDED__
#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>

class Matrix{
public:
  Matrix(unsigned cols, unsigned rows, bool initConnections);
  void value(unsigned col, unsigned row, float val) { _matrix[col][row] = val; }
  float value(unsigned col, unsigned row) { return _matrix[col][row]; }
  Matrix transpose();
  Matrix dotProduct(Matrix &m2);
  void sum(Matrix &m2);
  void show();
private:
  std::vector<std::vector <float> > _matrix;
};

#endif
