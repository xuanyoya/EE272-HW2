// double buffer implementation for Catapult HLS
#include "ac_channel.h"
#include "stencil_catapult.h"

template<typename T, int N>
struct chanStruct{
  T data[N];
 };

//FIFO implemented as shift registers
template<int ID,typename DTYPE,int NUM_REGS> 
void fifo(DTYPE din, DTYPE &dout) {
  static DTYPE regs[NUM_REGS];

#pragma hls_unroll yes
SHIFT:for(int i=NUM_REGS-1; i>=0; i--) {
    if (i==0) {
      regs[i] = din;
    } else {
      regs[i] = regs[i-1];
    }
 }

  dout = regs[NUM_REGS-1];
}



#pragma hls_design
#pragma hls_pipeline_init_interval 1
template <typename DTYPE, int C_I, int XY_I, int XY_O, int C_O, int WS>
void WRITE_BLOCK_INPUT(ac_channel<PackedStencil<DTYPE,C_I> > &din,
                      ac_channel<chanStruct<DTYPE,XY_I> > &dout_0,
                      ac_channel<chanStruct<DTYPE,XY_I> > &dout_1,
                      ac_channel<chanStruct<DTYPE,XY_I> > &dout_2,
                      ac_channel<chanStruct<DTYPE,XY_I> > &dout_3) {


#pragma hls_pipeline_init_interval 1
  WRITE: for (int p_idx=0; p_idx < XY_O; p_idx++) {
    for (int c_idx=0; c_idx < C_O; c_idx++) {
      chanStruct<DTYPE, XY_I> tmp_0;    //temporary array inside struct
      chanStruct<DTYPE, XY_I> tmp_1;    //temporary array inside struct
      chanStruct<DTYPE, XY_I> tmp_2;    //temporary array inside struct
      chanStruct<DTYPE, XY_I> tmp_3;    //temporary array inside struct  
      for (int y_idx = 0; y_idx < 0 + XY_I; y_idx++)
      {
        PackedStencil<DTYPE,C_I,1,1> column;
        column = din.read();
        tmp_0.data[y_idx] = column(0,0,0);
        tmp_1.data[y_idx] = column(1,0,0);
        tmp_2.data[y_idx] = column(2,0,0);
        tmp_3.data[y_idx] = column(3,0,0);
      } // for y_idx
    
      dout_0.write(tmp_0);//Memory channel write
      dout_1.write(tmp_1);//Memory channel write
      dout_2.write(tmp_2);//Memory channel write
      dout_3.write(tmp_3);//Memory channel write
    } // for c_idx
  } // for p_idx
}


#pragma hls_design
#pragma hls_pipeline_init_interval 1
template <typename DTYPE, int Y_I, int X_I, int Y_O, int X_O, int C_I, int K_O, int C_O, int WS>
void READ_BLOCK_INPUT(ac_channel<chanStruct<DTYPE,(Y_I+WS-1)*(X_I+WS-1)> > &din_0,
                      ac_channel<chanStruct<DTYPE,(Y_I+WS-1)*(X_I+WS-1)> > &din_1,
                      ac_channel<chanStruct<DTYPE,(Y_I+WS-1)*(X_I+WS-1)> > &din_2,
                      ac_channel<chanStruct<DTYPE,(Y_I+WS-1)*(X_I+WS-1)> > &din_3,
                     ac_channel<PackedStencil<DTYPE, C_I,1,1> > &dout){

/*reuse the input pixels in the double buffer when iterating through different kernels
and window locations*/
#pragma hls_pipeline_init_interval 1
  READ: for(int ro_idx = 0; ro_idx < Y_O; ro_idx++) {
    for (int co_idx=0; co_idx < X_O; co_idx++) {    
      for (int c_idx = 0; c_idx <C_O; c_idx++) {
        chanStruct<DTYPE,(Y_I+WS-1)*(X_I+WS-1)> tmp_0;    //temporary array inside struct
        chanStruct<DTYPE,(Y_I+WS-1)*(X_I+WS-1)> tmp_1;    //temporary array inside struct
        chanStruct<DTYPE,(Y_I+WS-1)*(X_I+WS-1)> tmp_2;    //temporary array inside struct
        chanStruct<DTYPE,(Y_I+WS-1)*(X_I+WS-1)> tmp_3;    //temporary array inside struct    
        tmp_0 = din_0.read();                       // Single Memory channel read
        tmp_1 = din_1.read();                       // Single Memory channel read
        tmp_2 = din_2.read();                       // Single Memory channel read
        tmp_3 = din_3.read();                       // Single Memory channel read   
        for (int wx_idx = 0; wx_idx < WS; wx_idx++) {
        for (int wy_idx = 0; wy_idx < WS; wy_idx++) {
        for (int k_idx = 0; k_idx < K_O; k_idx++) {
        for (int x_idx=0; x_idx < Y_I; x_idx++) {
        for (int y_idx=0; y_idx < X_I; y_idx++)
        {
          PackedStencil<DTYPE, C_I,1,1> dout_struct;
          dout_struct(tmp_0.data[(x_idx+wx_idx)* (X_I+WS-1) +  y_idx + wy_idx], 0, 0, 0, 0);
          dout_struct(tmp_1.data[(x_idx+wx_idx)* (X_I+WS-1) +  y_idx + wy_idx], 1, 0, 0, 0);
          dout_struct(tmp_2.data[(x_idx+wx_idx)* (X_I+WS-1) +  y_idx + wy_idx], 2, 0, 0, 0);
          dout_struct(tmp_3.data[(x_idx+wx_idx)* (X_I+WS-1) +  y_idx + wy_idx], 3, 0, 0, 0); 
          dout.write(dout_struct);
        
        } // for y_idx
        } // for x_idx
        } // for k_idx
        } // for wy_idx
        } // for wx_idx
      } // for c_idx
    } // for co_idx
  } //for ro_idx
}

/*Input double buffer.
Inputs are a stream of input pixels, outputs are a stream of PackedStencil of pixels.
PackedStencil is a data struct that pack multiple elements into a long word to
increase the port width and bandwidth.
*/
#pragma hls_design
#pragma hls_pipeline_init_interval 1
template <typename DTYPE, int Y_I, int X_I, int Y_O, int X_O, int C_I, int K_O, int C_O, int WS>
void double_buffer_input( 
                         ac_channel<PackedStencil<DTYPE,C_I> > &din, 
                         ac_channel<PackedStencil<DTYPE, C_I,1,1> > &dout) {

  // Four banks of memorie, since the PE array is 4 x 4.
  static ac_channel<chanStruct<DTYPE,(Y_I+WS-1)*(X_I+WS-1)> > shr_mem_0;//Static memory channel
  static ac_channel<chanStruct<DTYPE,(Y_I+WS-1)*(X_I+WS-1)> > shr_mem_1;//Static memory channel
  static ac_channel<chanStruct<DTYPE,(Y_I+WS-1)*(X_I+WS-1)> > shr_mem_2;//Static memory channel
  static ac_channel<chanStruct<DTYPE,(Y_I+WS-1)*(X_I+WS-1)> > shr_mem_3;//Static memory channel

  WRITE_BLOCK_INPUT<DTYPE, C_I, (Y_I+WS-1)*(X_I+WS-1), Y_O*X_O, C_O, WS>(din, shr_mem_0, shr_mem_1, shr_mem_2, shr_mem_3);
  READ_BLOCK_INPUT<DTYPE, Y_I, X_I, Y_O, X_O, C_I, K_O, C_O, WS>(shr_mem_0, shr_mem_1, shr_mem_2, shr_mem_3, dout);

}

#pragma hls_design
#pragma hls_pipeline_init_interval 1
template <typename DTYPE, int KI, int C_I, int K_I, int XY_O, int K_O, int C_O, int WS>
void WRITE_BLOCK_WEIGHTS(ac_channel<PackedStencil<DTYPE, KI, K_I> > &din,
                         ac_channel<chanStruct<PackedStencil<DTYPE, KI, 1>, C_I*K_O*C_O*WS*WS> > &dout_0,
                         ac_channel<chanStruct<PackedStencil<DTYPE, KI, 1>, C_I*K_O*C_O*WS*WS> > &dout_1,
                         ac_channel<chanStruct<PackedStencil<DTYPE, KI, 1>, C_I*K_O*C_O*WS*WS> > &dout_2,
                         ac_channel<chanStruct<PackedStencil<DTYPE, KI, 1>, C_I*K_O*C_O*WS*WS> > &dout_3) {
                             
#pragma hls_pipeline_init_interval 1
  WRITE: for(int p_idx = 0; p_idx < XY_O; p_idx++) {
    chanStruct<PackedStencil<DTYPE, KI, 1>, C_I*K_O*C_O*WS*WS> tmp_0;    //temporary array inside struct
    chanStruct<PackedStencil<DTYPE, KI, 1>, C_I*K_O*C_O*WS*WS> tmp_1;    //temporary array inside struct
    chanStruct<PackedStencil<DTYPE, KI, 1>, C_I*K_O*C_O*WS*WS> tmp_2;    //temporary array inside struct
    chanStruct<PackedStencil<DTYPE, KI, 1>, C_I*K_O*C_O*WS*WS> tmp_3;    //temporary array inside struct
    for (int k_idx = 0; k_idx < K_O; k_idx++) {
      for (int c_idx = 0; c_idx < C_O; c_idx++) {
        for (int wx_idx=0; wx_idx < WS*WS; wx_idx++) {
          for (int r_idx = 0; r_idx < 0 + C_I; r_idx++)
          {
            PackedStencil<DTYPE, KI, K_I> row;
            row     = din.read();
            tmp_0.data[k_idx*C_I*C_O*WS*WS + c_idx*C_I*WS*WS + wx_idx*C_I  + r_idx] = row.get_dim(0,0,0);
            tmp_1.data[k_idx*C_I*C_O*WS*WS + c_idx*C_I*WS*WS + wx_idx*C_I  + r_idx] = row.get_dim(1,0,0);
            tmp_2.data[k_idx*C_I*C_O*WS*WS + c_idx*C_I*WS*WS + wx_idx*C_I  + r_idx] = row.get_dim(2,0,0);
            tmp_3.data[k_idx*C_I*C_O*WS*WS + c_idx*C_I*WS*WS + wx_idx*C_I  + r_idx] = row.get_dim(3,0,0);
          } // for r_idx
        } // for wx_idx
      } //for c_idx
    } // for k_idx
    dout_0.write(tmp_0);//Memory channel write
    dout_1.write(tmp_1);//Memory channel write
    dout_2.write(tmp_2);//Memory channel write
    dout_3.write(tmp_3);//Memory channel write
  } // for p_idx
}


#pragma hls_design
#pragma hls_pipeline_init_interval 1
template <typename DTYPE, int KI, int K_I, int XY_I, int XY_O, int C_I, int K_O, int C_O, int WS>
void READ_BLOCK_WEIGHTS(ac_channel<chanStruct<PackedStencil<DTYPE,KI,1,1,1>, C_I*K_O*C_O*WS*WS> > &din_0,
                        ac_channel<chanStruct<PackedStencil<DTYPE,KI,1,1,1>, C_I*K_O*C_O*WS*WS> > &din_1,
                        ac_channel<chanStruct<PackedStencil<DTYPE,KI,1,1,1>, C_I*K_O*C_O*WS*WS> > &din_2,
                        ac_channel<chanStruct<PackedStencil<DTYPE,KI,1,1,1>, C_I*K_O*C_O*WS*WS> > &din_3,
                        ac_channel<PackedStencil<DTYPE, KI, K_I,1,1> > &dout){


//reuse the weights in the double buffer when looping through different image tiles.
#pragma hls_pipeline_init_interval 1
  READ: for(int p_idx = 0; p_idx < XY_O; p_idx++) {
    chanStruct<PackedStencil<DTYPE, KI, 1>,C_I*K_O*C_O*WS*WS> tmp_0;    //temporary array inside struct
    chanStruct<PackedStencil<DTYPE, KI, 1>,C_I*K_O*C_O*WS*WS> tmp_1;    //temporary array inside struct
    chanStruct<PackedStencil<DTYPE, KI, 1>,C_I*K_O*C_O*WS*WS> tmp_2;    //temporary array inside struct
    chanStruct<PackedStencil<DTYPE, KI, 1>,C_I*K_O*C_O*WS*WS> tmp_3;    //temporary array inside struct

    tmp_0 = din_0.read();                       // Single Memory channel read
    tmp_1 = din_1.read();                       // Single Memory channel read
    tmp_2 = din_2.read();                       // Single Memory channel read
    tmp_3 = din_3.read();                       // Single Memory channel read     
    for (int c_idx = 0; c_idx <C_O; c_idx++) {
      for (int wx_idx = 0; wx_idx < WS*WS; wx_idx++){
        for (int k_idx = 0; k_idx < K_O; k_idx++) {
          for (int r_idx = 0; r_idx < C_I; r_idx++)
          {
            PackedStencil<DTYPE, KI, K_I> dout_struct;
            dout_struct.set_dim(tmp_0.data[k_idx*C_I*C_O*WS*WS + c_idx*C_I*WS*WS + wx_idx*C_I + r_idx], 0, 0, 0);
            dout_struct.set_dim(tmp_1.data[k_idx*C_I*C_O*WS*WS + c_idx*C_I*WS*WS + wx_idx*C_I + r_idx], 1, 0, 0);
            dout_struct.set_dim(tmp_2.data[k_idx*C_I*C_O*WS*WS + c_idx*C_I*WS*WS + wx_idx*C_I + r_idx], 2, 0, 0);
            dout_struct.set_dim(tmp_3.data[k_idx*C_I*C_O*WS*WS + c_idx*C_I*WS*WS + wx_idx*C_I + r_idx], 3, 0, 0);
            dout.write(dout_struct);
          } // for r_idx
        } // for k_idx
      } // for wx_idx
    } // for c_idx
  } // for p_idx
}


/*Weight double buffer.
Inputs and outputs are a stream of PackedStencil of coefficients..
PackedStencil is a data struct that pack multiple elements into a long word to
increase the port width and bandwidth.
*/
#pragma hls_design
#pragma hls_pipeline_init_interval 1
template <typename DTYPE, int KI, int K_I, int XY_I, int XY_O, int C_I, int K_O, int C_O, int WS>
  void double_buffer_weights(ac_channel<PackedStencil<DTYPE, KI, K_I> > &din, 
                             ac_channel<PackedStencil<DTYPE, KI, K_I> > &dout) {

  // Four banks of memorie, since the PE array is 4 x 4.
  static ac_channel<chanStruct<PackedStencil<DTYPE, KI, 1,1,1>, C_I*K_O*C_O*WS*WS> > shr_mem_0;//Static memory channel
  static ac_channel<chanStruct<PackedStencil<DTYPE, KI, 1,1,1>, C_I*K_O*C_O*WS*WS> > shr_mem_1;//Static memory channel
  static ac_channel<chanStruct<PackedStencil<DTYPE, KI, 1,1,1>, C_I*K_O*C_O*WS*WS> > shr_mem_2;//Static memory channel
  static ac_channel<chanStruct<PackedStencil<DTYPE, KI, 1,1,1>, C_I*K_O*C_O*WS*WS> > shr_mem_3;//Static memory channel

  WRITE_BLOCK_WEIGHTS<DTYPE, KI, C_I, K_I, XY_O, K_O, C_O, WS>(din, shr_mem_0, shr_mem_1, shr_mem_2, shr_mem_3);
  READ_BLOCK_WEIGHTS<DTYPE, KI, K_I, XY_I, XY_O, C_I, K_O, C_O, WS>(shr_mem_0, shr_mem_1, shr_mem_2, shr_mem_3, dout);
}
