#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/display.hpp"


namespace graphics {
__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int Arows, int Acols, int Bcols);
} // namespace graphics