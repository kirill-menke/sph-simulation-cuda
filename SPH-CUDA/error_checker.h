#pragma once
#include <cuda_runtime.h>
#include <iostream>

__host__ void checkError(cudaError_t err)
{
	if (err != cudaSuccess)
	{
		// Print a human readable error message
		std::cout << cudaGetErrorString(err) << std::endl;
		exit(-1);
	}
}