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

#ifndef NET_H_
#define NET_H_

typedef struct Neuron
{
    double activation;
    double bias;
} Neuron;

typedef struct NET
{
    int layers;
    double error;
    int* NPL;
    Neuron** neurons;
    double** weights;
    double** backupWeights;
    double** deltaWeights;
    double** errors;
} NET;

void load_random_inputs(NET* Network);
void restore_weights(NET* Network);
void save_weights(NET* Network);

void print_entire_network(NET* Network);
void print_input_layer(NET* Network);
void print_output_layer(NET* Network);
void print_weights(NET* Network);
void print_expected_output(NET* Network, double* expectedOut);

NET* create_network(int LAYERS, int neuronsPerLayer[]);
int* set_NPL_array(int LAYERS, int neuronsPerLayer[]);
Neuron** create_neurons(int LAYERS, int neuronsPerLayer[]);
double** create_weights(int LAYERS, int neuronsPerLayer[]);
double** create_backup_weights(int LAYERS, int neuronsPerLayer[]);
double** create_delta_weights(int LAYERS, int neuronsPerLayer[]);
double** create_errors(int LAYERS, int neuronsPerLayer[]);

void initialize_network_values(NET* Network);
void initialize_neuron_values(Neuron** neurons, int LAYERS, int neuronsPerLayer[]);
void initialize_weight_values(double** weights, int LAYERS, int neuronsPerLayer[]);
void initialize_backup_weight_values(double** backupWeights, int LAYERS, int neuronsPerLayer[]);
void initialize_delta_weight_values(double** deltaWeights, int LAYERS, int neuronsPerLayer[]);
void initialize_error_values(double** errors, int LAYERS, int neuronsPerLayer[]);

void free_network(NET* Network);
void free_NPL_array(int* NPLArray);
void free_neurons(Neuron** neurons, int LAYERS);
void free_backup_weights(double** weights, int LAYERS);
void free_delta_weights(double** weights, int LAYERS);
void free_weights(double** weights, int LAYERS);
void free_errors(double** errors, int LAYERS);

#endif
