#include "../include/vector3d.cuh"
#include <math.h>

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

} // namespace graphics
