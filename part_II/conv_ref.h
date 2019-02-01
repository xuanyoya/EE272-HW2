//Reference model
//Xuan Yang 1/31/2019

#ifndef __CONV_REF__
#define __CONV_REF__
#define SC_INCLUDE_FX

#include "conv.h"

void conv_ref( DTYPE input[(OROW+W_SIZE-1)][(OCOL+W_SIZE-1)][C_NUM], 
               DTYPE weight[W_SIZE][W_SIZE][C_NUM][K_NUM], 
               DTYPE output[OROW][OCOL][K_NUM]){

  
  ROW:for (int i=0; i < OROW; ++i ) {
    COL:for (int j=0; j < OCOL; ++j) {
      NK: for (int k=0; k < K_NUM; ++k) {
        DTYPE tmp=0;
        ACC:for (int c=0; c < C_NUM; ++c) { 
          WR: for (int fx=0; fx < W_SIZE; fx++) {
            WC: for (int fy=0; fy < W_SIZE; fy++) {
              tmp += input[i+fx][j+fy][c] * weight[fx][fy][c][k];
            }
          }
        }
        output[i][j][k]= tmp;
      }
    }
  }
}
#endif

