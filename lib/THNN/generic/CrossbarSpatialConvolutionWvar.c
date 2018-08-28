#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/CrossbarSpatialConvolutionWvar.c"
#else

#include <math.h>

static void THNN_(unfolded_custom_padding_copy)(
          THTensor *finput,
          THTensor *input,
	  int padValue,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          int nInputPlane,
          int inputWidth,
          int inputHeight,
          int outputWidth,
          int outputHeight)
{
  // This function assumes that
  // kH*kW does not overflow an int
  // nInputPlane*kH*kW does not overflow a long
  // outputHeight*dH does not overflow a long
  // outputWidth*dW does not overflow a long

  long k;
  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);

#pragma omp parallel for private(k)
  for(k = 0; k < (long)nInputPlane*kH*kW; k++) {
    long nip = k / (kH*kW);
    long rest = k % (kH*kW);
    long kh = rest / kW;
    long kw = rest % kW;
    int x, y;
    long ix, iy;
    real *dst = finput_data + nip*((size_t)kH*kW*outputHeight*outputWidth) + kh*((size_t)kW*outputHeight*outputWidth) + kw*((size_t)outputHeight*outputWidth);
    real *src = input_data + nip*((size_t)inputHeight*inputWidth);
    if (padW > 0 || padH > 0) {
      long lpad,rpad;
      for(y = 0; y < outputHeight; y++) {
        iy = (long)y*dH - padH + kh;
        if (iy < 0 || iy >= inputHeight) {
          memset(dst+(size_t)y*outputWidth, padValue, sizeof(real)*outputWidth);
        } else {
          if (dW==1){
             ix = 0 - padW + kw;
             lpad = fmaxf(0,padW-kw);
             rpad = fmaxf(0,padW-(kW-kw-1));
             if (outputWidth-rpad-lpad <= 0) {
                memset(dst+(size_t)y*outputWidth, padValue, sizeof(real)*outputWidth);
             } else {
                if (lpad > 0) memset(dst+(size_t)y*outputWidth, padValue, sizeof(real)*lpad);
                memcpy(dst+(size_t)y*outputWidth+lpad, src+(size_t)iy*inputWidth+ix+lpad, sizeof(real)*(outputWidth-rpad-lpad));
                if (rpad > 0) memset(dst+(size_t)y*outputWidth + outputWidth - rpad, padValue, sizeof(real)*rpad);
             }
          }
          else{
            for (x=0; x<outputWidth; x++){
               ix = (long)x*dW - padW + kw;
               if (ix < 0 || ix >= inputWidth)
                 memset(dst+(size_t)y*outputWidth+x, padValue, sizeof(real)*1);
               else
                 memcpy(dst+(size_t)y*outputWidth+x, src+(size_t)iy*inputWidth+ix, sizeof(real)*(1));
            }
          }
        }
      }
    } else {
      for(y = 0; y < outputHeight; y++) {
        iy = (long)y*dH + kh;
        ix = 0 + kw;
        if (dW == 1)
           memcpy(dst+(size_t)y*outputWidth, src+(size_t)iy*inputWidth+ix, sizeof(real)*outputWidth);
        else{
          for (x=0; x<outputWidth; x++)
             memcpy(dst+(size_t)y*outputWidth+x, src+(size_t)iy*inputWidth+ix+(long)x*dW, sizeof(real)*(1));
         }
      }
    }
  }
}

static inline void THNN_(CrossbarSpatialConvolutionWvar_shapeCheck)(
  THTensor *input, THTensor *weight, THTensor *VarP, THTensor *VarM,
  int kH, int kW, int dH, int dW, int padH, int padW) {
  
  THArgCheck(kW > 0 && kH > 0, 9,
            "kernel size should be greater than zero, but got kH: %d kH: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THNN_ARGCHECK(weight->nDimension == 2, 5, weight, 
                "2D weight tensor expected, but got: %s");
  // check if every weight has VarP and VarM
  THArgCheck((THTensor_(nElement)(weight) == THTensor_(nElement)(VarP)) && 
	     (THTensor_(nElement)(weight) == THTensor_(nElement)(VarM)), 102,
	     "nElement of weight and VarP / VarM should be the same, but weight: %d VarP: %d, VarM: %d", 
	      THTensor_(nElement)(weight), 
	      THTensor_(nElement)(VarP),
	      THTensor_(nElement)(VarM));
  
  
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
static THTensor *THNN_(view_weight_CrossWvar2d)(THTensor *weight){
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

static void THNN_(CrossbarSpatialConvolutionWvar_updateOutput_frame)(
  THTensor *input,
  THTensor *output,
  THTensor *weight,
  THTensor *finput,
  THTensor *VarP,
  THTensor *VarM,
  int accumN,
  long nPsum,
  int padValue,
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
  THTensor *output2d;
  
  // Lowering convolution
  THNN_(unfolded_custom_padding_copy)(finput, input, padValue, kW, kH, dW, dH, padW, padH,
                       nInputPlane, inputWidth, inputHeight,
                       outputWidth, outputHeight);
  
  // Initialize output
  output2d = THTensor_(newWithStorage2d)(output->storage, output->storageOffset,
                                         nOutputPlane, -1,
                                         outputHeight*outputWidth, -1);
  
  // get pointer of real
  real *output2d_real = THTensor_(data)(output2d);
  real *finput_real = THTensor_(data)(finput);
  real *weight_real = THTensor_(data)(weight);
  real *VarP_real = THTensor_(data)(VarP);
  real *VarM_real = THTensor_(data)(VarM);
  
  // get parameters
  long nIn = weight->size[1];
  long nOutSpatial = outputHeight * outputWidth;
  
  // do the computation
  for (long i=0; i<nOutputPlane; i++) {
    for (long j=0; j<nOutSpatial; j++) {
      real output_temp = 0;
      for (long k=0; k<nPsum; k++) {
        // do the accumulation
        real psum = 0;
        for (int n=0; n<accumN; n++) {
          // multiplication
          real temp = finput_real[(k*accumN+n)*nOutSpatial+j] * weight_real[i*nIn+(k*accumN+n)];
// 	  printf("input idx: %ld, input: %.1f weight: %.1f temp %.1f ", (k*accumN+n)*nOutSpatial+j, finput_real[(k*accumN+n)*nOutSpatial+j], weight_real[i*nIn+(k*accumN+n)], temp);
          // variation modeling
          temp = (temp > 0)? 
                  temp + VarP_real[i*nIn+(k*accumN+n)] : temp + VarM_real[i*nIn+(k*accumN+n)];
          // accumulation
          psum += temp;
        }
	//quantize psum
	if (accumN == 1) {
		psum = (psum >= 0)? 1 : -1;
	}
	else {
		psum = round(psum/2)*2;
		//clamping
		psum = (psum > accumN)? accumN : psum;
		psum = (psum < (-1)*accumN)? (-1)*accumN : psum;
	}
        // update output_temp
        output_temp += psum;
      }
      // update result
      output2d_real[i*nOutSpatial+j] = output_temp;
    }
  }
  
  // free output2d
  THTensor_(free)(output2d);
}

	// old code for psum quantization
// 	printf("psum before quantize: %.1f ", psum); // zero padding is problem!!!!
//         // quantize psum
//         psum = (accumN == 1)? round(psum) : round(psum/2)*2;
//         // clamping
//         psum = (psum > accumN)? accumN : psum;
//         psum = (psum < (-1)*accumN)? (-1)*accumN : psum;
// 	printf("psum after quantize: %.1f\n", psum);

void THNN_(CrossbarSpatialConvolutionWvar_updateOutput)(
  THNNState *state,
  THTensor *input,
  THTensor *output,
  THTensor *weight,
  THTensor *finput,
  THTensor *VarP,
  THTensor *VarM,
  int accumN,
  int padValue
  int kW, int kH,
  int dW, int dH,
  int padW, int padH)
{
  weight = THNN_(view_weight_CrossWvar2d)(weight);
  
  THNN_(CrossbarSpatialConvolutionWvar_shapeCheck)
    (input, weight, VarP, VarM, kH, kW, dH, dW, padH, padW);
    
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
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);
    
    THNN_(CrossbarSpatialConvolutionWvar_updateOutput_frame)
      (input, output, weight, finput, 
       VarP, VarM, accumN, nPsum, padValue
       kW, kH, dW, dH, padW, padH,
       nInputPlane, inputWidth, inputHeight,
       nOutputPlane, outputWidth, outputHeight);
  }
  else {
    long T = input->size[0];
    long t;
    
    THTensor_(resize3d)(finput, T, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize4d)(output, T, nOutputPlane, outputHeight, outputWidth);
    
#pragma omp parallel for private(t)
    for (t = 0; t < T; t++) {
      THTensor *input_t = THTensor_(newSelect)(input, 0, t);
      THTensor *output_t = THTensor_(newSelect)(output, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);
      
      THNN_(CrossbarSpatialConvolutionWvar_updateOutput_frame)
      (input_t, output_t, weight, finput_t, 
       VarP, VarM, accumN, nPsum, padValue
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
