#include "../include/display.cuh"
#include "../include/constants.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/display.hpp"

namespace graphics {
__device__ __host__ void cudaRotate(Vector3D* vertex, double roll, double pitch, double yaw) {
    // Roll (rotation around X-axis)
    double cosR = cos(roll);
    double sinR = sin(roll);
    double y = vertex->y;
    double z = vertex->z;
    vertex->y = y * cosR - z * sinR;
    vertex->z = z * cosR + y * sinR;

    // Pitch (rotation around Y-axis)
    double cosP = cos(pitch);
    double sinP = sin(pitch);
    double x = vertex->x;
    z = vertex->z;
    vertex->x = x * cosP + z * sinP;
    vertex->z = z * cosP - x * sinP;

    // Yaw (rotation around Z-axis)
    double cosY = cos(yaw);
    double sinY = sin(yaw);
    x = vertex->x;
    y = vertex->y;
    vertex->x = x * cosY - y * sinY;
    vertex->y = y * cosY + x * sinY;
}

__device__ __host__ void cudaTranslate(Vector3D* vertex, double x, double y, double z) {
    vertex->x += x;
    vertex->y += y;
    vertex->z += z;
}

__device__ __host__ void cudaProject(const Vector3D* vertex, Vector2D* projectedVertex) {
    projectedVertex->x = (vertex->x * FOV) / vertex->z;
    projectedVertex->y = (vertex->y * FOV) / vertex->z;
}

__global__ void transformVerticesKernel(Vector3D* vertices, Vector2D* projectedVertices, int size, Vector3D rotation, Vector3D camera) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        Vector3D vertex = vertices[idx];

        // Rotate the vertex
        cudaRotate(&vertex, rotation.x, rotation.y, rotation.z);

        // Translate the vertex
        cudaTranslate(&vertex, camera.x, camera.y, -camera.z);

        // Project the transformed vertex
        cudaProject(&vertex, &projectedVertices[idx]);
    }
}
void Display::InitalizeCuda() {
  cudaMalloc((void**)&d_vertices, NUM_VERTICES * sizeof(Vector3D));
  cudaMalloc((void**)&d_projectedVertices, NUM_VERTICES * sizeof(Vector2D));
}
void Display::FreeCuda() {
    cudaFree(d_vertices);
    cudaFree(d_projectedVertices);
}
void Display::LaunchCuda() {
    // Copy vertices to device
    cudaMemcpy(d_vertices, vertices.data(), NUM_VERTICES * sizeof(Vector3D), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_VERTICES + threadsPerBlock - 1) / threadsPerBlock;
    transformVerticesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_vertices, d_projectedVertices, NUM_VERTICES, rotation, camera);

    // Copy projected vertices back to host
    cudaMemcpy(projectedVertices.data(), d_projectedVertices, NUM_VERTICES * sizeof(Vector2D), cudaMemcpyDeviceToHost);

    // Synchronize
    cudaDeviceSynchronize();
}
} // namespace graphics
