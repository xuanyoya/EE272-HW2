//
// Copyright 2003-2015 Mentor Graphics Corporation
//
// All Rights Reserved.
//
// THIS WORK CONTAINS TRADE SECRET AND PROPRIETARY INFORMATION WHICH IS THE PROPERTY OF 
// MENTOR GRAPHICS CORPORATION OR ITS LICENSORS AND IS SUBJECT TO LICENSE TERMS.
// 

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "conv_ref.h"
#include "conv.h"
using namespace std;


#define DEBUG
// Helper function to parse feature map data from CSV file
template<int ROW, int COL, int CH>
void parse_featuremap(const string &filename, DTYPE feature[ROW][COL][CH]){
  ifstream infile(filename);
  int row = 0;
  while(infile) {
    string s;
    if (!getline(infile, s)) break;
   
    istringstream ss(s);
    int col = 0;
    while(ss) {
      string cur;
      if (!getline(ss, cur, ',')) break;
      feature[col/COL][col%COL][row] = stof(cur);
      col++;
    }
    row++;
  }
}


// Helper function to parse weight data from CSV file
template<int W, int C, int K>
void parse_weight(const string &filename, DTYPE weight[W][W][C][K]){
  ifstream infile(filename);
  int row = 0;
  while(infile) {
    string s;
    if (!getline(infile, s)) break;
   
    istringstream ss(s);
    int col = 0;
    while(ss) {
      string cur;
      if (!getline(ss, cur, ',')) break;
      weight[col/W][col%W][row%C][row/C] = stof(cur);
      col++;
    }
    row++;
  }
}

/*Test the functionality of the golden model
  KNUM       ofmap channel number (kernel number)
  CNUM       ifmap channel number
  W_SIZE     window width and height
  OROW       output image height
  OCOL       output image width
  STRIDE     window stride
*/
template <int OROW, int OCOL, int CNUM, int KNUM, int W_SIZE, int STRIDE>
void test(string in_file, string weight_file, string out_file) 
{
  
  DTYPE input[(OROW*STRIDE+W_SIZE-1)][(OCOL*STRIDE+W_SIZE-1)][CNUM]; 
  DTYPE weight[W_SIZE][W_SIZE][CNUM][KNUM]; 
  DTYPE output_ref[OROW][OCOL][KNUM];
  DTYPE output[OROW][OCOL][KNUM];


  int errCnt = 0;

  parse_featuremap<OROW*STRIDE+W_SIZE-1, OCOL*STRIDE+W_SIZE-1, CNUM>(in_file, input);
  
  parse_weight<W_SIZE, CNUM, KNUM>(weight_file, weight);

  parse_featuremap<OROW, OCOL, KNUM>(out_file, output);

  conv_ref<OROW, OCOL, CNUM, KNUM, W_SIZE, STRIDE>(input, weight, output_ref);          

  //printf("\nCompare output\n\n"); 
  for (int k = 0; k < KNUM; k++) {
    for (int p = 0; p < OROW; p++ ){   
      for (int i = 0; i < OCOL; i++ ){   
        if(output_ref[p][i][k] != output[p][i][k]) {
          errCnt++;
          printf("output[%d][%d][%d] = %f, ref = %f\n",p, i, k, output[p][i][k], output_ref[p][i][k]); 
        }
      }  
    }
  }

  printf("There were %d errors\n",errCnt);
}

int main(int argc, char *argv[]) 
{
  printf("Testing simple layer\n");
  string suffix = "_simple.csv";
  test<simple.row, simple.col, simple.channel, simple.kernel, simple.window, simple.stride>(INFILE+suffix, WEIGHTFILE+suffix, OUTFILE+suffix);

  printf("\nTesting resnet0 layer\n");
  suffix = "_resnet0.csv";
  test<resnet0.row, resnet0.col, resnet0.channel, resnet0.kernel, resnet0.window, resnet0.stride>(INFILE+suffix, WEIGHTFILE+suffix, OUTFILE+suffix);

  printf("\nTesting resnet1 layer\n");
  suffix = "_resnet1.csv";
  test<resnet1.row, resnet1.col, resnet1.channel, resnet1.kernel, resnet1.window, resnet1.stride>(INFILE+suffix, WEIGHTFILE+suffix, OUTFILE+suffix);

  printf("\nTesting resnet2 layer\n");
  suffix = "_resnet2.csv";
  test<resnet2.row, resnet2.col, resnet2.channel, resnet2.kernel, resnet2.window, resnet2.stride>(INFILE+suffix, WEIGHTFILE+suffix, OUTFILE+suffix);

  printf("\nTesting resnet3 layer\n");
  suffix = "_resnet3.csv";
  test<resnet3.row, resnet3.col, resnet3.channel, resnet3.kernel, resnet3.window, resnet3.stride>(INFILE+suffix, WEIGHTFILE+suffix, OUTFILE+suffix);

  printf("\nTesting resnet4 layer\n");
  suffix = "_resnet4.csv";
  test<resnet4.row, resnet4.col, resnet4.channel, resnet4.kernel, resnet4.window, resnet4.stride>(INFILE+suffix, WEIGHTFILE+suffix, OUTFILE+suffix);

  return 0;
}

