#!/bin/bash

g++ Neuron.cpp Matrix.cpp Net.cpp main\ MNIST.cpp \
-lsfml-graphics -lsfml-window -lsfml-system -std=c++11 -o "Neural Network"
./Neural\ Network
