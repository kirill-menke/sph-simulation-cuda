#pragma once
#include <cuda_runtime.h>
#include <iostream>

void __inline__ __cudaSafeCall(cudaError err, const char *file, const int line) {
	if (cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString(err));
		getchar();
		exit(-1);
	}
}

#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
