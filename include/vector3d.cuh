#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/vector3d.hpp"
#include "../include/vector2d.hpp"
#include "../include/constants.hpp"

namespace graphics {
__device__ __host__ void cudaRotate(Vector3D* vertex, double roll, double pitch, double yaw);
__device__ __host__ void cudaTranslate(Vector3D* vertex, double x, double y, double z);
__device__ __host__ void cudaProject(const Vector3D* vertex, Vector2D* projectedVertex);
}  // namespace graphics
