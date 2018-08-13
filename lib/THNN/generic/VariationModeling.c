#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VariationModeling.c"
#else

void THNN_(VariationModeling_updateOutput)(
  THNNState *state,
  THTensor *output,
  THTensor *input,
  THTensor *ptable,
  int accumN,
  THTensor *ref)
{
  long dim = THTensor_(nDimension)(input);
  if (dim == 1) {
  }
  else if (dim == 2) {
    // get parameters
    long dim1_input = THTensor_(size)(input,0);
    long dim2_input = THTensor_(size)(input,1);
    long nRow_ptable = THTensor_(size)(ptable,0);
    long nCol_ptable = THTensor_(size)(ptable,1);
    long transitionWindow = (nCol_ptable - 1)/2;
    // resize and zero initialize output
    
    // get pointer of real
    
    // do the modeling
  }
  else if (dim == 3) {
  }
  else if (dim == 4) {
  }
}


#endif
