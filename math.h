/*
MIT License

Copyright (c) 2021 Michael Maldonado

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef MATH_H_
#define MATH_H_

#include "net.h"
#include "input.h"

void train_network(NET* Network, int epochs, int trainDataSize, double learningRate, double momentum);
double test_network(NET* Network, int trainDataSize, int testDataSize);

void simulate_network(NET* Network, int trainDataPos, bool isTesting, bool isTestingData, double learningRate, double momentum);

void forward_propagation(NET* Network);
void compute_output_error(NET* Network, double* expectedOut);
void backpropagate_errors(NET* Network);
void adjust_weights(NET* Network, double learningRate, double momentum);

double sigmoid_function(double input);

#endif
