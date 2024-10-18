#include "../include/display.cuh"
#include "../include/vector3d.cuh"
#include "../include/constants.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/display.hpp"

namespace graphics {
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
