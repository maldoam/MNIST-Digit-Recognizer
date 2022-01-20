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

#include "input.h"

void read_training_images()
{
    int i, j;

    imageBuff = (unsigned char*)malloc(sizeof(unsigned char) * IMG_SIZE);

    imageFile = fopen(TRAIN_IMG_PATH, "rb");

    fread(imageInfo, sizeof(int), 4, imageFile);

    for (i = 0; i < NUM_TRAIN_IMG; i++)
    {
        fread(imageBuff, sizeof(unsigned char), IMG_SIZE, imageFile);
        
        for (j = 0; j < IMG_SIZE; j++)
        {
            data.trainIMG[i][j] = (double)imageBuff[j] / (double)255;
        }
    }

    free(imageBuff);
    fclose(imageFile);
}


void read_training_labels()
{
    int i;

    labelBuff = (unsigned char*)malloc(sizeof(unsigned char));

    labelFile = fopen(TRAIN_LABEL_PATH, "rb");

    fread(labelInfo, sizeof(int), 2, labelFile);
    
    for (i = 0; i < NUM_TRAIN_IMG; i++)
    {
        fread(labelBuff, sizeof(unsigned char), 1, labelFile);

        data.trainLABEL[i] = (int)(*labelBuff);
    }

    free(labelBuff);
    fclose(labelFile);
}


void read_testing_images()
{
    int i, j;

    imageBuff = (unsigned char*)malloc(sizeof(unsigned char) * IMG_SIZE);

    imageFile = fopen(TEST_IMG_PATH, "rb");

    fread(imageInfo, sizeof(int), 4, imageFile);
    
    for (i = 0; i < NUM_TEST_IMG; i++)
    {
        fread(imageBuff, sizeof(unsigned char), IMG_SIZE, imageFile);

        for (j = 0; j < IMG_SIZE; j++)
        {
            data.testIMG[i][j] = (double)imageBuff[j] / (double)255;
        }  
    }

    free(imageBuff);
    fclose(imageFile);
}


void read_testing_labels()
{
    int i;

    labelBuff = (unsigned char*)malloc(sizeof(unsigned char));

    labelFile = fopen(TEST_LABEL_PATH, "rb");

    fread(labelInfo, sizeof(int), 2, labelFile);
    
    for (i = 0; i < NUM_TEST_IMG; i++)
    {
        fread(labelBuff, sizeof(unsigned char), 1, labelFile);

        data.testLABEL[i] = (int)(*labelBuff);
    }

    free(labelBuff);
    fclose(labelFile);
}


void load_mnist_data()
{
    read_training_images();
    read_training_labels();

    read_testing_images();
    read_testing_labels();
}


void load_single_image(NET* Network, int imageNum, int isTesting)
{
    int imageLabel = (isTesting == 1 ? data.testLABEL[imageNum] : data.trainLABEL[imageNum]);

    //Testing Image Selected
    if (isTesting == 1)
    {
        int j;
        for (j = 0; j < IMG_SIZE; j++)
        {
            Network -> neurons[0][j].activation = data.testIMG[imageNum][j];
        }
    }
    //Training Image Selected
    else
    {
        int i;
        for (i = 0; i < IMG_SIZE; i++)
        {
            Network -> neurons[0][i].activation = data.trainIMG[imageNum][i];
        }
    }

    // printf("Load %s_IMG[%d] label: %d\n", (isTesting == 1 ? "TEST" : "TRAIN"), imageNum, imageLabel);
}


void print_image(int imageNum, int isTesting)
{
    int i;
    int imageLabel = (isTesting == 1 ? data.testLABEL[imageNum] : data.trainLABEL[imageNum]);

    for (i = 0; i < 784; i++)
    {   
        //Testing Image Selected
        if (isTesting == 1)
        {
            if (data.testIMG[imageNum][i] == 0.0)
            {
                printf("   ");
            }
            else
            {
                printf("%1.1f", data.testIMG[imageNum][i]);
            }
        }
        //Training Image Selected
        else
        {
            if (data.trainIMG[imageNum][i] == 0.0)
            {
                printf("   ");
            }
            else
            {
                printf("%1.1f", data.trainIMG[imageNum][i]);
            }
        }

        if ((i + 1) % 28 == 0)
        {
            printf("\n");
        }
    }

    printf("%s_IMG[%d] label: %d\n", (isTesting == 1 ? "TEST" : "TRAIN"), imageNum, imageLabel);
}


void label_to_expected_out(double* expectedOut, int imageNum, int isTesting)
{
    int i; 
    int imageLabel = (isTesting == 1 ? data.testLABEL[imageNum] : data.trainLABEL[imageNum]);

    for (i = 0; i < 10; i++)
    {
        if (i != imageLabel)
        {
            expectedOut[i] = (double)0;
        }
        else if (i == imageLabel)
        {
            expectedOut[i] = (double)1;
        }
    }
}
