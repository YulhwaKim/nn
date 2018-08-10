#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/CrossbarCompute.c"
#else

// void THNN_(CrossbarCompute_updateAddBuffer)(
// 	THNNState *state,
// 	THTensor *input,
// 	THTensor *addBuffer) 
// {
// 	long nframe = THTensor_(size)(input,0);
// 	long nElement = THTensor_(nElement)(addBuffer);
// 	if (nElement != nframe) {
// 		THTensor_(resize1d)(addBuffer,nframe);
// 		THTensor_(fill)(addBuffer,1.0);
// 	}
// } 

void THNN_(CrossbarCompute_updateOutput)(
	THNNState *state,
	THTensor *output,
	THTensor *input,
	THTensor *weight,
	int accumN)
{
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
		THTensor_(resize3d)(output, nframe, nOut, nPsum);
		if (THTensor_(nElement)(output) != nElement) {
			THTensor_(zero)(output);
		}
		// get pointer of real
		real *output_real = THTensor_(data)(output);
		real *input_real = THTensor_(data)(input);
		real *weight_real = THTensor_(data)(weight);
		// do the computation
		for(long i=0; i<nframe ; i++) {
			for(long j=0; j<nOut ; j++) {
				for(long k=0; k<nPsum ; k++) {
					// do the accumulation
					// THTensor temp = 0;
					real temp = 0;
					for(long n=0; n<accumN; n++) {
						temp += input_real[i*nIn+(k*accumN+n)] * weight_real[(k*accumN+n)*nOut+j];
					}
					// update result
					output_real[i*(nOut*nPsum)+j*nPsum+k] = temp;
				}
			}
		}
	}
}

#endif
