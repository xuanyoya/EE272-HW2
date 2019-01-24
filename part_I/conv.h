//
// Copyright 2003-2015 Mentor Graphics Corporation
//
// All Rights Reserved.
//
// THIS WORK CONTAINS TRADE SECRET AND PROPRIETARY INFORMATION WHICH IS THE PROPERTY OF 
// MENTOR GRAPHICS CORPORATION OR ITS LICENSORS AND IS SUBJECT TO LICENSE TERMS.
// 


#ifndef _GLOBAL_SIMPLE_H
#define _GLOBAL_SIMPLE_H
#define SC_INCLUDE_FX


/*
#define KNUM       16  //ofmap channel number (kernel number)
#define CNUM       16  //ifmap channel number
#define W_SIZE     3   //window width and height
#define OROW       16  //output image height
#define OCOL       8   //output image width
*/
#define INFILE      "input"
#define WEIGHTFILE  "weight"
#define OUTFILE     "output"


struct layer {
  int row;      //ofmap image height
  int col;      //ofmap image width
  int channel;  //ifmap channel number
  int kernel;   //ofmap channel number (kernel number)
  int window;   //window width and height
  int stride;   //window stride 
};


// Example layers
constexpr struct layer simple = {16, 8, 16, 16, 3, 1};
constexpr struct layer resnet0 = {56, 56, 64, 64, 3, 1};
constexpr struct layer resnet1 = {28, 28, 128, 128, 3, 1};
constexpr struct layer resnet2 = {14, 14, 256, 256, 3, 1};
constexpr struct layer resnet3 = {28, 28, 128, 128, 3, 2};
constexpr struct layer resnet4 = {14, 14, 256, 256, 3, 2};

// Define the data type (precision) used for the algorithm
typedef float DTYPE;

#endif

