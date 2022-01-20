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

#ifndef INPUT_H_
#define INPUT_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#include "net.h"

#define TRAIN_IMG_PATH "./data/train-images.idx3-ubyte"
#define TRAIN_LABEL_PATH "./data/train-labels.idx1-ubyte"

#define TEST_IMG_PATH "./data/t10k-images.idx3-ubyte"
#define TEST_LABEL_PATH "./data/t10k-labels.idx1-ubyte"

#define IMG_SIZE 784

#define NUM_TRAIN_IMG 60000
#define NUM_TEST_IMG 10000

int imageInfo[4];
int labelInfo[2];

unsigned char* imageBuff;
unsigned char* labelBuff;

FILE* labelFile;
FILE* imageFile;

typedef struct MNIST
{
    double trainIMG[NUM_TRAIN_IMG][IMG_SIZE];
    double testIMG[NUM_TEST_IMG][IMG_SIZE];
    int trainLABEL[NUM_TRAIN_IMG];
    int testLABEL[NUM_TEST_IMG];
} MNIST;

MNIST data;

void read_training_images();
void read_training_labels();

void read_testing_images();
void read_testing_labels();

void convert_training_images();
void convert_training_labels();

void convert_testing_images();
void convert_testing_labels();

void load_mnist_data();

void load_single_image(NET* Network, int imageNum, int isTesting);
void print_image(int imageNum, int isTesting);
void label_to_expected_out(double* expectedOut, int imageNum, int isTesting);

#endif
