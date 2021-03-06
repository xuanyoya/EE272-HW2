// Top-level design. 
// Xuan Yang 1/31/2019
#ifndef _CONV_H
#define _CONV_H

#include "double_buffer.h"
#include "conv.h"
#include "pe_template.h"

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/cat.hpp>

#define ARRAY_DIMENSION 16
#define REPEAT(x) BOOST_PP_REPEAT(ARRAY_DIMENSION, x, 0)

/*
The systolic array is 4 X 4. unrolling C_I (=4) channels amd K_I (=4) kernels.
The input and output of systolic array are streams of input, weight and output.
*/
#pragma hls_design 
#pragma hls_pipeline_init_interval 2
template<typename DTYPE, int K_II, int K_I, int Y_I, int X_I, int Y_O, int X_O, int C_I, int K_O, int C_O, int WS>
void systolic_array(ac_channel<PackedStencil<DTYPE, C_I, 1, 1> > &input, 
                    ac_channel<PackedStencil<DTYPE, K_II, K_I, 1> > &weight, 
                    ac_channel<PackedStencil<DTYPE, K_II, K_I, 1> > &output) {

  const int XY_I = X_I * Y_I;
  const int XY_O = X_O * Y_O;

  // C_I x K_I PE array
  static pe_template<DTYPE, K_II> pe[C_I+1][K_I+1];

  // local buffers to store partial output 
  // There are four of them because K_I = 4 
  #define OUT_TILE_INIT(z,i,unused)\
    PackedStencil<DTYPE, K_II, 1, 1> BOOST_PP_CAT(out_tile_,i)[XY_I*K_O];   
  REPEAT(OUT_TILE_INIT)

  /*
  the registers that used for relaying input and output in horizonal and vertical directions respectively.
  PE[i][j] fetch input data from register in_tmp[i+1][j], at next cycle forward the data to in_tmp[i+1][j+1]
  PE[i][j] fetch output data from register out_tmp[i][j+1], at next cycle forward the data to out_tmp[i+1][j+1]
  */
  DTYPE in_tmp[C_I+1][K_I+1];
  PackedStencil<DTYPE, K_II, 1, 1> out_tmp[C_I+1][K_I+1];
  // loop over image tiles
  xy_o: for (int p = 0; p < XY_O; ++p) {
  // loop over channel tile
  co: for (int c_idx = 0; c_idx < C_O; ++c_idx) {
  // loop over filter window
  winx: for (int wx_idx = 0; wx_idx < WS; ++wx_idx) {
  winy: for (int wy_idx = 0; wy_idx < WS; ++wy_idx) {
  // loop over kernel tiles
  ko: for (int k_idx = 0; k_idx < K_O; ++k_idx) {
  // loop inside each image tile
  xy_i: for (int step = 0; step < K_I+C_I+XY_I-1; ++step) {
        static PackedStencil<DTYPE,K_II, K_I> w_tile[C_I];
  
        // filling phase for systolic array, put data into local registers 
        if (step < C_I) {            
          PackedStencil<DTYPE,K_II, K_I> w_row = weight.read();
          w_tile[step] = w_row;
          /*#ifndef __SYNTHESIS__
          for (int col = 0; col<K_I; col++) {
            printf("weight=%d on row  %d, col %d\n", w_row(0,col,0,0), step, col);
          }
          #endif*/
  
        }
  
        /* read input from the output stream of the double buffer,
        push input to fifos, and read input from fifos into local registers*/
        PackedStencil<DTYPE, C_I,1,1> in_col;
        if (step < XY_I) {        
          in_col = input.read();
          /*#ifndef __SYNTHESIS__
          for (int row = 0; row<C_I; row++) {
            printf("input=%d on row  %d, col %d\n", in_col(row,0,0,0), step, row);
          }
          #endif*/
  
        }
 
        // The local registers serve data to the first column of PE array. 
        PackedStencil<DTYPE, C_I,1,1> input_buf;

        /* A trianglar shape of FIFOs, used for skewing the array front,
        such that the right input data comes to the right PE at the right timing.*/
        #define INPUT_FIFO_BODY(z,i,unused) \
          BOOST_PP_CAT(DTYPE input_fifo_, i); \
          fifo<60000+i,DTYPE,i+1>( in_col(i ,0,0), BOOST_PP_CAT(input_fifo_, i));\
          input_buf( BOOST_PP_CAT(input_fifo_, i), i ,0,0,0);
        REPEAT(INPUT_FIFO_BODY)
  
        /*#ifndef __SYNTHESIS__
        printf("starting step %d - input %d %d %d %d\n", step, input_fifo_0,input_fifo_1,input_fifo_2,input_fifo_3);
        #endif*/
  

        // local registers to store partial output
        #define TMP_ROW_BODY(z,i,unused) \
          PackedStencil<DTYPE, K_II, 1, 1, 1> BOOST_PP_CAT(tmp_row_, i);
        REPEAT(TMP_ROW_BODY)

        if (step < XY_I) {
          if(c_idx == 0 && wx_idx == 0 && wy_idx == 0) {
                #pragma hls_unroll yes           
                for (int sk = 0; sk < K_II; sk++) {
                  #define TMP_ROW_BODY_INIT(z,i,unused) \
                    BOOST_PP_CAT(tmp_row_, i)(0,sk,0,0,0);
                  REPEAT(TMP_ROW_BODY_INIT)

                }
          } else {
            #define TMP_ROW_OUT(z,i,unused) \
              BOOST_PP_CAT(tmp_row_, i) = BOOST_PP_CAT(out_tile_, i)[k_idx*XY_I + step];
            REPEAT(TMP_ROW_OUT)

          }
        }
       
        /* A trianglar shape of FIFOs, used for skewing the array front, 
        such that the right partial output data come to the right PE at the right timing*/ 
        PackedStencil<DTYPE, K_II, K_I,1> output_buf;
          
        #define TMP_FIFO_BODY(z,i,unused) \
          PackedStencil<DTYPE, K_II> BOOST_PP_CAT(tmp_fifo_,i);\
          fifo<90000+i,PackedStencil<DTYPE,K_II>, i+1>( BOOST_PP_CAT(tmp_row_,i), BOOST_PP_CAT(tmp_fifo_,i) );\
          output_buf.set_dim( BOOST_PP_CAT(tmp_fifo_, i), i,0,0);
        REPEAT(TMP_FIFO_BODY)
      
          /*#ifndef __SYNTHESIS__
          printf("starting step %d - partial result %d %d %d %d\n", step, tmp_fifo_0,tmp_fifo_1,tmp_fifo_2,tmp_fifo_3);
          #endif*/
   
          //initialize the input registers in the first column 
          #pragma hls_unroll yes
          INIT_IN: for(int i = 0; i < C_I; ++i) {
            in_tmp[i+1][0] = input_buf(i,0,0);
          }
    
          //initialize the output registers in the first row 
          #pragma hls_unroll yes
          INIT_OUT: for(int j = 0; j < K_I; ++j) {
            out_tmp[0][j+1] = output_buf.get_dim(j, 0, 0);
          }
    
          // perform the a matrix multiplication in a systolic fashion 
          #pragma hls_unroll yes
          COL: for (int j=0; j < K_I; ++j) {
            #pragma hls_unroll yes
            ROW: for (int i=0; i < C_I; ++i) {
              PackedStencil<DTYPE, K_II> weight_value = w_tile[i].get_dim(j,0,0);
              pe[i][j].exec(in_tmp[i+1][j], out_tmp[i][j+1], weight_value, in_tmp[i+1][j+1], out_tmp[i+1][j+1]);
            } //ROW
          } //COL
  
          /* A trianglar shape of FIFOs, used for skewing as well, 
          such that the right output data are collected at the right timing*/ 
          PackedStencil<DTYPE, K_II, K_I> output_row;
    
          #define FIFO_WRITE_BODY(z,i,unused)\
            PackedStencil<DTYPE, K_II> BOOST_PP_CAT(sys_array_out_,i) = out_tmp[C_I][i+1];\
            PackedStencil<DTYPE, K_II> BOOST_PP_CAT(output_fifo_,i); \
            fifo<0+i,PackedStencil<DTYPE, K_II>, K_I-i>( BOOST_PP_CAT(sys_array_out_,i), BOOST_PP_CAT(output_fifo_,i) );\
            output_row.set_dim( BOOST_PP_CAT(output_fifo_,i), i,0,0); 
          REPEAT(FIFO_WRITE_BODY)

          /*#ifndef __SYNTHESIS__
            printf("ending step %d - output %d %d %d %d\n", step, output_fifo_0,output_fifo_1,output_fifo_2,output_fifo_3);
          #endif*/
    
          // output row if one has completed
          if (step >= K_I+C_I-1) {
            #define OUTPUT_ROW_BODY(z,i,unused)\
              BOOST_PP_CAT(out_tile_,i)[k_idx*XY_I+step-(K_I+C_I-1)] = BOOST_PP_CAT(output_fifo_,i);
            REPEAT(OUTPUT_ROW_BODY)

            if (c_idx==C_O-1 && wx_idx == WS-1 && wy_idx == WS-1) {
              output.write(output_row);
            }
          }
    } //STEPS
    } //K_O
    } //WS
    } //WS
    } //C_O
    } //XY_O
}

/*
The top level design.
Inputs are streams of input, weight.
Outputs is a stream of output.
This design consists a input double buffer, a weight double buffer, and a systolic array.
Input and weight data are reused inside double buffers, and streamed to systolic array.
Output data are accumulated inside systolic array, and streamed out.
*/

#pragma hls_design top
#pragma hls_pipeline_init_interval 1
void conv(ac_channel<PackedStencil<DTYPE,CI_NUM> > &input, 
          ac_channel<PackedStencil<DTYPE, KII, KI_NUM> > &weight, 
          ac_channel<PackedStencil<DTYPE, KII, KI_NUM> > &output) {
   
  static ac_channel<PackedStencil<DTYPE, CI_NUM,1,1> > input_stream; 
  static ac_channel<PackedStencil<DTYPE, KII, KI_NUM,1> > weight_stream;                     


  double_buffer_input<DTYPE, OROW_I, OCOL_I, OROW_O, OCOL_O, CI_NUM, KO_NUM, CO_NUM, W_SIZE>(input, input_stream);

  double_buffer_weights<DTYPE, KII, KI_NUM, OROW_I*OCOL_I, OROW_O*OCOL_O, CI_NUM, KO_NUM, CO_NUM, W_SIZE>(weight, weight_stream);

  systolic_array<DTYPE, KII, KI_NUM, OROW_I, OCOL_I, OROW_O, OCOL_O, CI_NUM, KO_NUM, CO_NUM, W_SIZE>(input_stream, weight_stream, output);
 
}
#endif
