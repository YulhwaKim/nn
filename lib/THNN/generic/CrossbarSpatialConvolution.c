#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/CrossbarSpatialConvolution.c"
#else

static inline void THNN_(CrossbarSpatialConvolution_shapeCheck)(
  THTensor *input, THTensor *weight,
  int kH, int kW, int dH, int dW, int padH, int padW) {
  
  THArgCheck(kW > 0 && kH > 0, 9,
            "kernel size should be greater than zero, but got kH: %d kH: %d", kH, kW);
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
  long nPsum,
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
  THTensor *output3d;
  
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
        for (int n=0; n<accumN; n++) {
          temp += finput_real[(k*accumN+n)*nOutSpatial+j] * weight_real[i*nIn+(k*accumN+n)];
        }
        // update result
        output3d_real[i*(nOutSpatial*nPsum)+j*nPsum+k] = temp;
      }
    }
  }
  
  // free output3d
  THTensor_(free)(output3d);
}

void THNN_(CrossbarSpatialConvolution_updateOutput)(
  THNNState *state,
  THTensor *input,
  THTensor *output,
  THTensor *weight,
  THTensor *finput,
  int accumN,
  int kW, int kH,
  int dW, int dH,
  int padW, int padH)
{
  weight = THNN_(view_weight_Cross2d)(weight);
  
  THNN_(CrossbarSpatialConvolution_shapeCheck)
    (input, weight, kH, kW, dH, dW, padH, padW);
    
  input = THTensor_(newContiguous)(input);
  int ndim = input->nDimension;
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;
  
  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }
  
  // get parameters
  long nInputPlane = input->size[dimf];
  long inputHeight = input->size[dimh];
  long inputWidth = input->size[dimw];
  long nOutputPlane = weight->size[0];
  long outputHeight = (inputHeight + 2*padH -kH) / dH + 1;
  long outputWidth = (inputWidth + 2*padW - kW) / dW + 1;
  long nPsum = weight->size[1] / accumN;  
  //Check if nPsum is valid
  THArgCheck(nPsum > 0 && weight->size[1] == nPsum * accumN, 101,
            "Number of input per convolution should be divisible by accumN, but we got number of input: %ld, accumN: %d, nPsum: %ld",
             weight->size[1], accumN, nPsum);
  
  // do the computation
  if (input->nDimension == 3) {
    THTensor_(resize2d)(finput, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize4d)(output, nOutputPlane, outputHeight, outputWidth, nPsum);
    
    THNN_(CrossbarSpatialConvolution_updateOutput_frame)
      (input, output, weight, finput, accumN, nPsum,
       kW, kH, dW, dH, padW, padH,
       nInputPlane, inputWidth, inputHeight,
       nOutputPlane, outputWidth, outputHeight);
  }
  else {
    long T = input->size[0];
    long t;
    
    THTensor_(resize3d)(finput, T, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize5d)(output, T, nOutputPlane, outputHeight, outputWidth, nPsum);
    
#pragma omp parallel for private(t)
    for (t = 0; t < T; t++) {
      THTensor *input_t = THTensor_(newSelect)(input, 0, t);
      THTensor *output_t = THTensor_(newSelect)(output, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);
      
      THNN_(CrossbarSpatialConvolution_updateOutput_frame)
      (input_t, output_t, weight, finput_t, accumN, nPsum,
       kW, kH, dW, dH, padW, padH,
       nInputPlane, inputWidth, inputHeight,
       nOutputPlane, outputWidth, outputHeight);
      
      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
      THTensor_(free)(finput_t);
    }
  }
  
  THTensor_(free)(input);
  THTensor_(free)(weight);
}

      
#endif
