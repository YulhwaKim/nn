#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VariationModeling.c"
#else

// libraries for random number generation
#include <stdlib.h>
#include <time.h>

void THNN_(VariationModeling_updateOutput)(
  THNNState *state,
  THTensor *output,
  THTensor *input,
  THTensor *ptable,
  int accumN,
  THTensor *ref) // ref is for debugging
{
  // get parameters
  long nElement = THTensor_(nElement)(input);
  long nRow_ptable = THTensor_(size)(ptable,0);
  long nCol_ptable = THTensor_(size)(ptable,1);
  long transitionWindow = (nCol_ptable - 1)/2;
  
  // resize output
  THTensor_(resizeAs)(output, input);
  
  // get pointer of real
  real *output_real = THTensor_(data)(output);
  real *input_real = THTensor_(data)(input);
  real *ptable_real = THTensor_(data)(ptable);
  real *ref_real = THTensor_(data)(ref);
  
  // do the modeling
  printf('test\n');
  for(long i=0; i<nElement; i++) {
    // STEP1. get data and row index of probability table
    int value = (int)input_real[i];
    int rowIdx = (value + accumN) / 2;
    // STEP2. generate reference point
    real refpoint = ref_real[i];
    // real refpoint = rand()/(float)RNAD_MAX;
    // STEP3. find the column index of probability table and change the data
    for(unsigned int j=0; j<nCol_ptable; j++) {
      real prob = ptable_real[rowIdx*nCol_ptable + j];
      if(((prob > 0) && (prob > refpoint)) || (j==nCol_ptable-1)) {
        output_real[i] = (real)value + 2*(j - transitionWindow);
        break;
      }
    }
  }
}


#endif
