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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "net.h"


void save_weights(NET* Network)
{
    int i, j;
    for (i = 0; i < (Network -> layers - 1); i++)
    {   
        for (j = 0; j < (Network -> NPL[i] * Network -> NPL[i + 1]); j++)
        {
            Network -> backupWeights[i][j] = Network -> weights[i][j];
        }
    }
}


void restore_weights(NET* Network)
{
    int i, j;
    for (i = 0; i < (Network -> layers - 1); i++)
    {   
        for (j = 0; j < (Network -> NPL[i] * Network -> NPL[i + 1]); j++)
        {
            Network -> weights[i][j] = Network -> backupWeights[i][j];
        }
    }
}


void load_random_inputs(NET* Network)
{
    int j;
    for (j = 0; j < Network -> NPL[0]; j++)
    {
        //Random double 0 -> 1
        Network -> neurons[0][j].activation = (double)rand() / (double)RAND_MAX;
    }
}

void print_output_layer(NET* Network)
{
    int i;

    printf("\nOUTPUT:\n");
    for (i = 0; i < Network -> NPL[Network -> layers - 1]; i++)
    {
        printf("L:%d N:%-3d A:%-2lf B:%-2lf\n", 
            (Network -> layers - 1), i, 
            Network -> neurons[Network -> layers - 1][i].activation, 
            Network -> neurons[Network -> layers - 1][i].bias);
    }
    printf("\n");
}


void print_input_layer(NET* Network)
{
    int i;

    printf("\nINPUT:\n\n");
    for (i = 0; i < Network -> NPL[0]; i++)
    {
        printf("L:%d N:%-3d A:%-2lf B:%-2lf\n", 
            (0), i, 
            Network -> neurons[0][i].activation, 
            Network -> neurons[0][i].bias);
    }
}


void print_weights(NET* Network)
{
    int i, j;
    int neuronFrom, neuronTo;

    printf("\nWEIGHTS:\n\n");
    for (i = 0; i < Network -> layers - 1; i++)
    {
        for (j = 0; j < (Network -> NPL[i] * Network -> NPL[i + 1]); j++)
        {
            neuronFrom = (j / Network -> NPL[i + 1]);
            neuronTo   = (j % Network -> NPL[i + 1]);

            if (j > 0 && neuronTo == 0)
            {
                printf("-\n");
            }

            printf("( L:%d N:%-3d  --[W:%10lf]->  L:%d N:%-2d )\n", 
                i, neuronFrom, 
                Network -> weights[i][j], 
                (i + 1), neuronTo);
        }
    }
}


void print_entire_network(NET* Network)
{
    int i, j;

    printf("\nNEURONS:\n\n");
    for (i = 0; i < Network -> layers; i++)
    {
        for (j = 0; j < Network -> NPL[i]; j++)
        {
            printf("L:%d N:%-3d A:%-2lf B:%-2lf\n", 
                i, j, 
                Network -> neurons[i][j].activation, 
                Network -> neurons[i][j].bias);
        }

        printf("\n");
    }

    int neuronFrom, neuronTo;

    printf("\nWEIGHTS:\n\n");
    for (i = 0; i < Network -> layers - 1; i++)
    {
        for (j = 0; j < (Network -> NPL[i] * Network -> NPL[i + 1]); j++)
        {
            neuronFrom = (j / Network -> NPL[i + 1]);
            neuronTo   = (j % Network -> NPL[i + 1]);

            if (j > 0 && neuronTo == 0)
            {
                printf("-\n");
            }

            printf("( L:%d N:%-3d  --[W:%10lf]->  L:%d N:%-2d )\n", 
                i, neuronFrom, 
                Network -> weights[i][j], 
                (i + 1), neuronTo);
        }

        printf("***\n");
    }
}


void print_expected_output(NET* Network, double* expectedOut)
{
    int i; 

    printf("\nExpected Output:\n");
    
    for (i = 0; i < Network -> NPL[Network -> layers - 1]; i++)
    {
        printf("L:%d N:%-3d A:%-2lf\n", (Network -> layers - 1), i, expectedOut[i]);
    }
}


NET* create_network(int LAYERS, int neuronsPerLayer[])
{
    NET* temp = (NET*)malloc(sizeof(NET));
    
    temp -> layers = LAYERS;
    temp -> error = 0.0;

    temp -> NPL = set_NPL_array(LAYERS, neuronsPerLayer);
    temp -> neurons = create_neurons(LAYERS, neuronsPerLayer);
    
    temp -> weights = create_weights(LAYERS, neuronsPerLayer);
    temp -> backupWeights = create_backup_weights(LAYERS, neuronsPerLayer);
    temp -> deltaWeights = create_delta_weights(LAYERS, neuronsPerLayer);
    
    temp -> errors = create_errors(LAYERS, neuronsPerLayer);

    return temp;
}


int* set_NPL_array(int LAYERS, int neuronsPerLayer[])
{
    int x;
    int* temp;

    temp = (int*)malloc(LAYERS * sizeof(int));
    for (x = 0; x < LAYERS; x++)
    {
        temp[x] = neuronsPerLayer[x];
    }

    return temp;
}


Neuron** create_neurons(int LAYERS, int neuronsPerLayer[])
{
    int x;
    Neuron** temp;

    temp = (Neuron**)malloc(LAYERS * sizeof(Neuron*));
    for (x = 0; x < LAYERS; x++)
    {
        temp[x] = (Neuron*)malloc(neuronsPerLayer[x] * sizeof(Neuron));
    }

    return temp;
}


double** create_weights(int LAYERS, int neuronsPerLayer[])
{
    int x;
    double** temp;

    temp = (double**)malloc((LAYERS - 1) * sizeof(double*));
    for (x = 0; x < LAYERS - 1; x++)
    {
        printf("WEIGHTS: layer[%d] %d\n", x, neuronsPerLayer[x] * neuronsPerLayer[x + 1]);
        int numWeights = (neuronsPerLayer[x] * neuronsPerLayer[x + 1]);
        
        temp[x] = (double*)malloc(numWeights * sizeof(double));
    }

    return temp;
}


double** create_backup_weights(int LAYERS, int neuronsPerLayer[])
{
    int x;
    double** temp;

    temp = (double**)malloc((LAYERS - 1) * sizeof(double*));
    for (x = 0; x < LAYERS - 1; x++)
    {
        int numWeights = (neuronsPerLayer[x] * neuronsPerLayer[x + 1]);
        
        temp[x] = (double*)malloc(numWeights * sizeof(double));
    }

    return temp;
}


double** create_delta_weights(int LAYERS, int neuronsPerLayer[])
{
    int x;
    double** temp;

    temp = (double**)malloc((LAYERS - 1) * sizeof(double*));
    for (x = 0; x < LAYERS - 1; x++)
    {
        int numWeights = (neuronsPerLayer[x] * neuronsPerLayer[x + 1]);
        
        temp[x] = (double*)malloc(numWeights * sizeof(double));
    }

    return temp;
}


double** create_errors(int LAYERS, int neuronsPerLayer[])
{
    int x;
    double** temp;

    temp = (double**)malloc(LAYERS * sizeof(double*));
    for (x = 0; x < LAYERS; x++)
    {
        temp[x] = (double*)malloc(neuronsPerLayer[x] * sizeof(double));
    }

    return temp;
}


void initialize_network_values(NET* Network)
{
    initialize_neuron_values(Network -> neurons, Network -> layers, Network -> NPL);
    
    initialize_weight_values(Network -> weights, Network -> layers, Network -> NPL);
    initialize_backup_weight_values(Network -> backupWeights, Network -> layers, Network -> NPL);
    initialize_delta_weight_values(Network -> deltaWeights, Network -> layers, Network -> NPL);
    
    initialize_error_values(Network -> errors, Network -> layers, Network -> NPL);
}


void initialize_neuron_values(Neuron** neurons, int LAYERS, int* neuronsPerLayer)
{
    int i, j;
    for (i = 0; i < LAYERS; i++)
    {
        for (j = 0; j < neuronsPerLayer[i]; j++)
        {
            neurons[i][j].activation = 0;
            neurons[i][j].bias = 0;
        }
    }
}


void initialize_weight_values(double** weights, int LAYERS, int* neuronsPerLayer)
{
    int i, j, neuronsOut, neuronsIn;
    double xInit;

    for (i = 0; i < LAYERS - 1; i++)
    {   
        neuronsOut = (neuronsPerLayer[i] * neuronsPerLayer[i + 1]);

        if (i == 0)
        {
            neuronsIn = 0;
        }
        else 
        {
            neuronsIn = (neuronsPerLayer[i - 1] * neuronsPerLayer[i]);
        }

        xInit = sqrt(6) / sqrt(neuronsIn + neuronsOut);
        
        printf("xavierInit LAYER[%d -> %d]: (%lf <=> %lf)\n", i, (i + 1), -(xInit), xInit);
        
        for (j = 0; j < (neuronsPerLayer[i] * neuronsPerLayer[i + 1]); j++)
        {
            weights[i][j] = (xInit + xInit) * ((double)rand() / (double)RAND_MAX) - xInit;
        }
    }

    printf("\n");
}


void initialize_backup_weight_values(double** backupWeights, int LAYERS, int neuronsPerLayer[])
{
    int i, j;
    for (i = 0; i < LAYERS - 1; i++)
    {   
        for (j = 0; j < (neuronsPerLayer[i] * neuronsPerLayer[i + 1]); j++)
        {
            backupWeights[i][j] = 0.0;
        }
    }
}


void initialize_delta_weight_values(double** deltaWeights, int LAYERS, int neuronsPerLayer[])
{
    int i, j;
    for (i = 0; i < LAYERS - 1; i++)
    {   
        for (j = 0; j < (neuronsPerLayer[i] * neuronsPerLayer[i + 1]); j++)
        {
            deltaWeights[i][j] = 0.0;
        }
    }
}


void initialize_error_values(double** errors, int LAYERS, int neuronsPerLayer[])
{
    int i, j;
    for (i = 0; i < LAYERS; i++)
    {
        for (j = 0; j < neuronsPerLayer[i]; j++)
        {
            errors[i][j] = 0.0;
        }
    }
}


void free_network(NET* Network)
{
    free_NPL_array(Network -> NPL);
    free_neurons(Network -> neurons, Network -> layers);
    free_backup_weights(Network -> backupWeights, Network -> layers);
    free_delta_weights(Network -> deltaWeights, Network -> layers);
    free_weights(Network -> weights, Network -> layers);
    free_errors(Network -> errors, Network -> layers);

    free(Network);
}


void free_backup_weights(double** backupWeights, int LAYERS)
{
    int i;
    for (i = 0; i < LAYERS - 1; i++)
    {
        free(backupWeights[i]);
    }

    free(backupWeights);
}


void free_delta_weights(double** deltaWeights, int LAYERS)
{
    int i;
    for (i = 0; i < LAYERS - 1; i++)
    {
        free(deltaWeights[i]);
    }

    free(deltaWeights);
}


void free_NPL_array(int* NPLArray)
{
    free(NPLArray);
}


void free_neurons(Neuron** neurons, int LAYERS)
{
    int i;
    for (i = 0; i < LAYERS; i++)
    {
        free(neurons[i]);
    }

    free(neurons);
}


void free_weights(double** weights, int LAYERS)
{
    int i;
    for (i = 0; i < LAYERS - 1; i++)
    {
        free(weights[i]);
    }

    free(weights);
}


void free_errors(double** errors, int LAYERS)
{
    int i;
    for (i = 0; i < LAYERS; i++)
    {
        free(errors[i]);
    }

    free(errors);
}

