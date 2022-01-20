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
#include <math.h>
#include <stdbool.h>

#include "math.h"


void simulate_network(NET* Network, int dataPos, bool isTesting, bool isTestingData, double learningRate, double momentum)
{
    double expectedOut[Network -> NPL[Network -> layers - 1]];

    load_single_image(Network, dataPos, isTestingData);
    label_to_expected_out(expectedOut, dataPos, isTestingData);
    
    forward_propagation(Network);
    compute_output_error(Network, expectedOut);
    
    if (!isTesting)
    {
        backpropagate_errors(Network);
        adjust_weights(Network, learningRate, momentum);
    }
    // print_output_layer(Network);
}

double test_network(NET* Network, int trainDataSize, int testDataSize)
{
    double trainingError = 0.0;
    double testingError = 0.0;

    int trainingSet, testingSet;

    for (trainingSet = 0; trainingSet < 1000; trainingSet++)
    {
        simulate_network(Network, trainingSet, true, false, 0.0, 0.0);
        trainingError += Network -> error;
    }
    // printf("  Finished testing trainingSet\n");

    for (testingSet = 0; testingSet < 1000; testingSet++)
    {
        simulate_network(Network, testingSet, true, true, 0.0, 0.0);
        testingError += Network -> error;
    }
    // printf("  Finished testing testingSet\n");

    printf("NMSE is %0.3f on Training Set and %0.3f on Test Set\n", trainingError, testingError);

    return testingError;
}


void train_network(NET* Network, int epochs, int trainDataSize, double learningRate, double momentum)
{
    int epoch, trainingSet;
    for (epoch = 0; epoch < epochs; epoch++)
    {
        // printf("  Training Epoch[%d]\n", epoch);

        for (trainingSet = 0; trainingSet < 1000; trainingSet++)
        {
            simulate_network(Network, trainingSet, false, false, learningRate, momentum);
        }
    }
}


void forward_propagation(NET* Network)
{
    int currentLayer, currentNeuron, currentWeight;
    int startWeightPos, endWeightPos, neuronFrom;
    double weightedSum = 0;

    for (currentLayer = 0; currentLayer < (Network -> layers - 1); currentLayer++)
    {
        for (currentNeuron = 0; currentNeuron < Network -> NPL[currentLayer + 1]; currentNeuron++)
        {
            startWeightPos = (currentNeuron * Network -> NPL[currentLayer]);
            endWeightPos = (startWeightPos + Network -> NPL[currentLayer]);

            for (currentWeight = startWeightPos; currentWeight < endWeightPos; currentWeight++)
            {
                neuronFrom = (currentWeight % Network -> NPL[currentLayer]);

                weightedSum += Network -> neurons[currentLayer][neuronFrom].activation * Network -> weights[currentLayer][currentWeight];
            }

            weightedSum += Network -> neurons[currentLayer + 1][currentNeuron].bias;

            Network -> neurons[currentLayer + 1][currentNeuron].activation = sigmoid_function(weightedSum);
        }
    }
}

void compute_output_error(NET* Network, double* expectedOut)
{
    int i;
    double currentActivation = 0.0;
    double currentError = 0.0;

    Network -> error = 0.0;

    for (i = 0; i < Network -> NPL[Network -> layers - 1]; i++)
    {
        currentActivation = Network -> neurons[Network -> layers - 1][i].activation;
        currentError = expectedOut[i] - currentActivation;

        Network -> errors[Network -> layers - 1][i] = (currentActivation * (1 - currentActivation) * currentError);
        Network -> error += (pow(currentError, 2) / 2);
    }

    // printf("      Total Network Error: %lf\n", Network -> error);

}


void backpropagate_errors(NET* Network)
{
    int currentLayer, lowerNeuron, upperNeuron, currentWeight;
    double currentActivation = 0.0;
    double currentError = 0.0;

    for (currentLayer = (Network -> layers - 1); currentLayer > 0; currentLayer--)
    {
        for (lowerNeuron = 0; lowerNeuron < Network -> NPL[currentLayer - 1]; lowerNeuron++)
        {
            currentActivation = Network -> neurons[currentLayer - 1][lowerNeuron].activation;
            currentError = 0.0;

            for (upperNeuron = 0; upperNeuron < Network -> NPL[currentLayer]; upperNeuron++)
            {
                // printf("currentLayer: %d, lowerNeuron: %d, upperNeuron: %d\n", currentLayer, lowerNeuron, upperNeuron);
                currentWeight = (upperNeuron * Network -> NPL[currentLayer]) + lowerNeuron;
                currentError += Network -> weights[currentLayer - 1][currentWeight] * Network -> errors[currentLayer][upperNeuron];
            }

            Network -> errors[currentLayer - 1][lowerNeuron] = (currentActivation * (1 - currentActivation) * currentError);
        }
    }
}


void adjust_weights(NET* Network, double learningRate, double momentum)
{
    int currentLayer, upperNeuron, lowerNeuron, dWeightPos;
    double currentActivation = 0.0;
    double currentError = 0.0;
    double currentDeltaWeight = 0.0;

    for (currentLayer = 1; currentLayer < Network -> layers; currentLayer++)
    {
        for (upperNeuron = 0; upperNeuron < Network -> NPL[currentLayer]; upperNeuron++)
        {
            for (lowerNeuron = 0; lowerNeuron < Network -> NPL[currentLayer - 1]; lowerNeuron++)
            {
                dWeightPos = (lowerNeuron) + (upperNeuron * Network -> NPL[currentLayer]);

                currentActivation = Network -> neurons[currentLayer - 1][lowerNeuron].activation;
                currentError = Network -> errors[currentLayer][upperNeuron];
                currentDeltaWeight = Network -> deltaWeights[currentLayer - 1][dWeightPos];

                Network -> weights[currentLayer - 1][dWeightPos] = (learningRate * currentError * currentActivation) + (momentum * currentDeltaWeight);
                Network -> deltaWeights[currentLayer - 1][dWeightPos] = (learningRate * currentError * currentActivation);
            }
        }
    }
}


double sigmoid_function(double input)
{
    return 1 / (1 + exp(-input));
}

