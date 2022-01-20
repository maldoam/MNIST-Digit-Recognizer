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
#include <time.h>
#include <stdbool.h>

#include "net.h"
#include "math.h"
#include "input.h"


int main(int argc, char** argv)
{
    srand(time(NULL));

    load_mnist_data();

    int LAYERS = 4;
    int neuronsPerLayer[] = {IMG_SIZE, 1000, 1000, 10};
    
    int epochs = 1;
    double learningRate = 0.5;
    double momentum = 0.0;
    
    double totalTestingError = 0.0;
    double minTestError = 99999999.9;

    bool stopFlag = false;

    NET* completedNetwork;

    completedNetwork = create_network(LAYERS, neuronsPerLayer);

    initialize_network_values(completedNetwork);

    
    while (stopFlag == false)
    {
        // printf("Before 'train_network'\n");
        train_network(completedNetwork, epochs, NUM_TRAIN_IMG, learningRate, momentum);
        // printf("Before 'test_network'\n");
        totalTestingError = test_network(completedNetwork, NUM_TRAIN_IMG, NUM_TEST_IMG);

        if (totalTestingError < minTestError)
        {
            printf(" -- Saving Weights\n");
            minTestError = totalTestingError;
            save_weights(completedNetwork);
        }

        else if (totalTestingError > (1.2 * minTestError))
        {
            printf(" -- Stopping Training and Restoring Weights\n");
            stopFlag = true;
            restore_weights(completedNetwork);
        }
    }
    
    // Uncomment to test, comment the other part (while)
    /*
    double expectedOut[completedNetwork -> NPL[completedNetwork -> layers - 1]];
    int i;
    int imageNum = 50;

    print_image(imageNum, false);

    for (i = 0; i < 1000; i++)
    {
        printf("\nITERATION: %d\n", i);
        load_single_image(completedNetwork, imageNum, false);
        label_to_expected_out(expectedOut, imageNum, false);

        printf("forward_propagation\n");
        forward_propagation(completedNetwork);

        print_output_layer(completedNetwork);
        printf("compute_output_error\n");

        compute_output_error(completedNetwork, expectedOut);

        printf("backpropagate_errors\n");
        backpropagate_errors(completedNetwork);
        
        printf("adjust_weights\n");
        adjust_weights(completedNetwork, learningRate, momentum);

        //print_weights(completedNetwork);
    }
    */

    free_network(completedNetwork);

    return 0;
}
