//
// Copyright 2003-2015 Mentor Graphics Corporation
//
// All Rights Reserved.
//
// THIS WORK CONTAINS TRADE SECRET AND PROPRIETARY INFORMATION WHICH IS THE PROPERTY OF 
// MENTOR GRAPHICS CORPORATION OR ITS LICENSORS AND IS SUBJECT TO LICENSE TERMS.
// 

#ifndef __CONV_REF__
#define __CONV_REF__
#define SC_INCLUDE_FX

#include "conv.h"

/*
The golden model of one CONV layer. 
ifmaps: input[height][width][channel] 
weight: weight[window_height][window_width][channel][kernel]
ofmaps: output[height][width][kernel]
*/
template<int OROW, int OCOL, int CNUM, int KNUM, int W_SIZE, int STRIDE>
void conv_ref( DTYPE input[(OROW*STRIDE+W_SIZE-1)][(OCOL*STRIDE+W_SIZE-1)][CNUM],
               DTYPE weight[W_SIZE][W_SIZE][CNUM][KNUM],
               DTYPE output[OROW][OCOL][KNUM])
{
  // Your code starts here  

}
#endif

