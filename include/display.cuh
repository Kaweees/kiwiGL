#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/display.hpp"

namespace graphics {
__global__ void transformVerticesKernel(Vector3D* vertices, Vector2D* projectedVertices, int size, Vector3D rotation, Vector3D camera);
} // namespace graphics