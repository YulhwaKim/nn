#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/CrossbarLinearWvar.c"
#else

#include <math.h>

void THNN_(CrossbarLinearWvar_updateOutput)(
	THNNState *state,
	THTensor *output,
	THTensor *input,
	THTensor *weight,
	THTensor *VarP,
	THTensor *VarM,
	int accumN)
{
	// check if every weight has VarP and VarM
	THArgCheck((THTensor_(nElement)(weight) == THTensor_(nElement)(VarP)) && 
		   (THTensor_(nElement)(weight) == THTensor_(nElement)(VarM)), 102,
			"nElement of weight and VarP / VarM should be the same, but weight: %d VarP: %d, VarM: %d", 
		    THTensor_(nElement)(weight), 
		    THTensor_(nElement)(VarP),
		    THTensor_(nElement)(VarM));
	
	long dim = THTensor_(nDimension)(input);
	if (dim == 1) {
		THError("Lazy Yulhwa did not prepare for the case that the dimension of input is 1!");
	}
	else if (dim == 2){
		// get parameters
		long nframe = THTensor_(size)(input,0);
		long nIn = THTensor_(size)(input,1);
		long nOut = THTensor_(size)(weight,0);
		long nElement = THTensor_(nElement)(output);
		long nPsum = nIn / accumN;
		// Check nPsum condition
		THArgCheck(nPsum * accumN == nIn, 101,
			"nIn should be divisible by accumN, but got nIn: %d accumN: %d", nIn, accumN);
		// resize and zero initialize output
		THTensor_(resize2d)(output, nframe, nOut);
		if (THTensor_(nElement)(output) != nElement) {
			THTensor_(zero)(output);
		}
		// get pointer of real
		real *output_real = THTensor_(data)(output);
		real *input_real = THTensor_(data)(input);
		real *weight_real = THTensor_(data)(weight);
		real *VarP_real = THTensor_(data)(VarP);
		real *VarM_real = THTensor_(data)(VarM);
		// do the computation
		for(long i=0; i<nframe ; i++) {
			for(long j=0; j<nOut ; j++) {
				real output_temp = 0;
				for(long k=0; k<nPsum ; k++) {
					// do the accumulation
					real psum = 0;
					for(long n=0; n<accumN; n++) {
						// multiplication
						real temp = input_real[i*nIn+(k*accumN+n)] * weight_real[j*nIn+(k*accumN+n)];
						// variation modeling
						temp = (temp > 0)? 
							temp + VarP_real[j*nIn+(k*accumN+n)] : temp + VarM_real[j*nIn+(k*accumN+n)];
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
				// update output
				output_real[i*nOut+j] = output_temp;
			}
		}
	}
}

					// old code for psum quantization
// 					// quantize psum
// 					psum = (accumN ==1)? round(psum) : round(psum/2)*2;
// 					// clamping
// 					psum = (psum > accumN)? accumN : psum;
// 					psum = (psum < (-1)*accumN)? (-1)*accumN : psum;

#endif
