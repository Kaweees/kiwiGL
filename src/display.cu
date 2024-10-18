#include "../include/display.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/display.hpp"

namespace graphics {
void Display::InitalizeCuda() {
  cudaMalloc((void**)&d_frameBuffer, displayMode.w * displayMode.h * sizeof(uint32_t));
}
void Display::FreeCuda() {
    cudaFree(d_frameBuffer);
}
} // namespace graphics
