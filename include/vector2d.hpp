#pragma once

#include <cmath>
#include <iostream>

namespace graphics {
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
}  // namespace graphics