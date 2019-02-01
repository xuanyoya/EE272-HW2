// PE class
// Xuan Yang 1/31/2019

#ifndef _PE_TEMPLATE_H
#define _PE_TEMPLATE_H

#include "conv.h"

// PE
template<typename DTYPE, int KI>
class pe_template{
  private:
    DTYPE x_reg;
    PackedStencil<DTYPE, KI, 1, 1> y_reg;
  public:
    void exec(DTYPE &x_in, PackedStencil<DTYPE, KI, 1, 1> &y_in, PackedStencil<DTYPE, KI, 1, 1> &w, DTYPE &x_out, PackedStencil<DTYPE, KI, 1, 1> &y_out) {
        x_out = x_reg;
        y_out = y_reg;
        x_reg = x_in;
        y_reg = y_in;
        COMP: for (int i = 0; i < KI; i++) {
            DTYPE tmp = x_reg * w(i, 0, 0) + y_reg(i, 0, 0);
            y_reg(tmp, i, 0, 0, 0);
        }
    }
};

#endif
