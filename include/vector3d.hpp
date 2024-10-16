#pragma once

#include <cmath>
#include <iostream>

#include "../include/vector2d.hpp"

using std::sqrt;

namespace graphics {
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
  Vector2D project() const;

  // Translate the vector
  void translate(double x, double y, double z);

  // Scale the vector
  void scale(double x, double y, double z);

  // Rotate the vector
  void rotate(double x, double y, double z);

  // Rotate the vector around the x-axis
  void rotateX(double theta);

  // Rotate the vector around the y-axis
  void rotateY(double theta);

  // Rotate the vector around the z-axis
  void rotateZ(double theta);
};
}  // namespace graphics
