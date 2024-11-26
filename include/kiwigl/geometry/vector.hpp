#pragma once

#include <cmath>

using std::sqrt;

#include "../core/constants.hpp"

namespace kiwigl {
// Represents a two-dimensional vector
class Vector2D {
  public:
  // Constructor to initialize memory
  Vector2D() : x(0), y(0) {}
  Vector2D(double x, double y) : x(x), y(y) {}

  // Destructor to free the memory allocated
  ~Vector2D() = default;

  // The x-coordinate of the vector
  double x;
  // The y-coordinate of the vector
  double y;

  // Translate the vector
  void translate(double x, double y) {
    this->x += x;
    this->y += y;
  }
};

// Represents a three-dimensional vector
class Vector3D {
  public:
  // Constructor to initialize memory
  Vector3D() : x(0), y(0), z(0) {}
  Vector3D(double x, double y, double z) : x(x), y(y), z(z) {}

  // Destructor to free the memory allocated
  ~Vector3D() = default;

  // The x-coordinate of the vector
  double x;
  // The y-coordinate of the vector
  double y;
  // The z-coordinate of the vector
  double z;

  // Project the vector onto a 2D plane
  Vector2D project() const { return Vector2D((x * FOV) / z, (y * FOV) / z); }

  // Translate the vector
  void translate(double x, double y, double z) {
    this->x += x;
    this->y += y;
    this->z += z;
  }

  // Scale the vector
  void scale(double x, double y, double z) {
    this->x *= x;
    this->y *= y;
    this->z *= z;
  }

  // Rotate the vector
  void rotate(double roll, double pitch, double yaw) {
    rotateX(roll);
    rotateY(pitch);
    rotateZ(yaw);
  }

  // Rotate the vector around the x-axis (roll)
  void rotateX(double theta) {
    double y_new = y * cos(theta) - z * sin(theta);
    double z_new = y * sin(theta) + z * cos(theta);
    y = y_new;
    z = z_new;
  }

  // Rotate the vector around the y-axis (pitch)
  void rotateY(double theta) {
    double x_new = x * cos(theta) - z * sin(theta);
    double z_new = x * sin(theta) + z * cos(theta);
    x = x_new;
    z = z_new;
  }

  // Rotate the vector around the z-axis (yaw)
  void rotateZ(double theta) {
    double x_new = x * cos(theta) - y * sin(theta);
    double y_new = x * sin(theta) + y * cos(theta);
    x = x_new;
    y = y_new;
  }
};
}  // namespace kiwigl
