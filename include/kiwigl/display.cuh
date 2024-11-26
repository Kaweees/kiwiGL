#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "constants.hpp"
#include "display.cuh"
#include "display.hpp"

namespace kiwigl {
__device__ __host__ void cudaRotate(
    Vector3D* vertex, double roll, double pitch, double yaw) {
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

__device__ __host__ void cudaTranslate3D(
    Vector3D* vertex, double x, double y, double z) {
  vertex->x += x;
  vertex->y += y;
  vertex->z += z;
}

__device__ __host__ void cudaTranslate2D(Vector2D* vertex, double x, double y) {
  vertex->x += x;
  vertex->y += y;
}

__device__ __host__ void cudaScale(
    Vector3D* vertex, double x, double y, double z) {
  vertex->x *= x;
  vertex->y *= y;
  vertex->z *= z;
}

__device__ __host__ void cudaProject(
    const Vector3D* vertex, Vector2D* projectedVertex) {
  projectedVertex->x = (vertex->x * FOV) / vertex->z;
  projectedVertex->y = (vertex->y * FOV) / vertex->z;
}

__global__ void transformVerticesKernel(Face* faces, Vector3D* vertices,
    Triangle* projectedTriangles, Vector3D rotation, Vector3D camera, int width,
    int height, int numFaces) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numFaces) return;

  // Process one face per thread
  Face face = faces[idx];
  for (int j = 0; j < 3; j++) {
    // Transform the vertices
    Vector3D vertex = vertices[face.vertexIndices[j] - 1];

    // Rotate the vertex
    cudaRotate(&vertex, rotation.x, rotation.y, rotation.z);

    // Translate the vertex
    cudaTranslate3D(&vertex, camera.x, camera.y, -camera.z);

    // Scale the vertex
    cudaScale(&vertex, 1.01, 1.01, 1.01);

    // Project the transformed vertex
    cudaProject(&vertex, &projectedTriangles[idx].points[j]);

    // Translate the projected vertices to the center of the screen
    cudaTranslate2D(&projectedTriangles[idx].points[j], width / 2, height / 2);
  }
}

void Display::InitalizeCuda() {
  // Allocate memory on the device
  cudaError_t err =
      cudaMalloc((void**)&d_faces, mesh.faces.size() * sizeof(Face));
  if (err != cudaSuccess) {
    fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), __FILE__,
        __LINE__);
    exit(EXIT_FAILURE);
  }
  err =
      cudaMalloc((void**)&d_vertices, mesh.vertices.size() * sizeof(Vector3D));
  if (err != cudaSuccess) {
    fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), __FILE__,
        __LINE__);
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc(
      (void**)&d_projectedTriangles, mesh.faces.size() * sizeof(Triangle));
  if (err != cudaSuccess) {
    fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), __FILE__,
        __LINE__);
    exit(EXIT_FAILURE);
  }
}
void Display::FreeCuda() {
  if (d_faces != nullptr) {
    cudaFree(d_faces);
    d_faces = nullptr;
  }
  if (d_vertices != nullptr) {
    cudaFree(d_vertices);
    d_vertices = nullptr;
  }
  if (d_projectedTriangles != nullptr) {
    cudaFree(d_projectedTriangles);
    d_projectedTriangles = nullptr;
  }
}
void Display::LaunchCuda(int width, int height) {
  // Copy faces to device
  cudaMemcpy(d_faces, mesh.faces.data(), mesh.faces.size() * sizeof(Face),
      cudaMemcpyHostToDevice);
  cudaMemcpy(d_vertices, mesh.vertices.data(),
      mesh.vertices.size() * sizeof(Vector3D), cudaMemcpyHostToDevice);

  // Launch kernel
  dim3 threadsPerBlock(256, 1, 1);
  dim3 blocksPerGrid(
      (mesh.faces.size() + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
  transformVerticesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_faces,
      d_vertices, d_projectedTriangles, rotation, camera, width, height,
      mesh.faces.size());

  // Synchronize to ensure all operations are complete
  cudaDeviceSynchronize();

  // Copy projected triangles back to host
  cudaMemcpy(projectedTriangles.data(), d_projectedTriangles,
      mesh.faces.size() * sizeof(Triangle), cudaMemcpyDeviceToHost);

  // Check for CUDA errors
  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(cudaStatus),
        __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
}
}  // namespace kiwigl
