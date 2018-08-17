#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/CrossbarSpatialConvolution.c"
#else

static inline void THNN_(CrossbarSpatialConvolution_shapeCheck)(
  THTensor *input, THTensor *weight,
  int kH, int kW, int dH, int dW, int padH, int padW) {
  
  THArgCheck(kW > 0 && kH > 0, 9,
            "kernel size should be greater than zero, but got kH: %d kH: %d", kH, KW);
  THArgCheck(dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THNN_ARGCHECK(weight->nDimension == 2, 5, weight, 
                "2D weight tensor expected, but got: %s");
  
  int ndim = input->nDimension;
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;
  
  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }
  
  THNN_ARGCHECK(ndim == 3 || ndim == 4, 2, input,
               "3D or 4D input tensor expected but got: %s");
  
  long nInputPlane = weight->size[1] / (kH * kW);
  long inputHeight = input->size[dimh];
  long inputWidth = input->size[dimw];
  long nOutputPlane = weight->size[0];
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  long outputWidth = (inputWidth + 2*padW - kW) / dW + 1;
  
  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%d x %d x %d). "
            "Calculated output size: (%d x %d x %d). Output size is too small",
            nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight, outputWidth);
  
  THNN_CHECK_DIM_SIZE(input, ndim, dimf, nInputPlane);
}

// Convert 4D weight into 2D weight
static THTensor *THNN_(view_weight_Cross2d)(THTensor *weight){
  weight = THTensor_(newContiguous)(weight);
  if (weight->nDimension == 4) {
    long s1 = weight->size[0];
    long s2 = weight->size[1] * weight->size[2] * weight->size[3];
    THTensor *old_weight = weight;
    weight = THTensor_(newWithStorage2d)(weight->storage, weight->storageOffset,
                                         s1, -1, s2, -1);
    THTensor_(free)(old_weight);
  }
  return weight;
}

static void THNN_(CrossbarSpatialConvolution_updateOutput_frame)(
  THTensor *input,
  THTensor *output,
  THTensor *weight,
  THTensor *finput,
  int accumN,
  int nPsum,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  long nInputPlane,
  long inputWidth,
  long inputHeight,
  long nOutputPlane,
  long outputWidth,
  long outputHeight)
{
  long i;
  THTensor *output2d;
  
  // Lowering convolution
  THNN_(unfolded_copy)(finput, input, kW, kH, dW, dH, padW, padH,
                       nInputPlane, inputWidth, inputHeight,
                       outputWidth, outputHeight);
  
  // Initialize output
  output3d = THTensor_(newWithStorage3d)(output->storage, output->storageOffset,
                                         nOutputPlane, -1,
                                         outputHeight*outputWidth, -1,
                                         nPsum, -1);
  
  // get pointer of real
  real *output3d_real = THTensor_(data)(output3d);
  real *finput_real = THTensor_(data)(finput);
  real *weight_real = THTensor_(data)(weight);
  
  // get parameters
  long nIn = nInputPlane * inputWidth * inputHeight;
  long nOutSpatial = outputHeight * outputWidth;
  
  // do the computation
  for (long i=0; i<nOutputPlane; i++) {
    for (long j=0; j<nOutSpatial; j++) {
      for (long k=0; k<nPsum; k++) {
        // do the accumulation
        real temp = 0;
        for (long n=0; n<accumN; n++) {
          temp += finput_real[(k*accumN+n)*nOutSpatial+j] * weight_real[i*nIn+(k*accumN+n)];
        }
        // update result
        output3d_real[i*(nOutSpatial*nPsum)+j*Psum+k] = temp;
      }
    }
  }
  
  // free output3d
  THTensor_(free)(output3d);
}
      
#endif
