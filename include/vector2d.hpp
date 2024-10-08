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

  // Overload the addition operator
  Vector2D operator+(const Vector2D& vec) const {
    return Vector2D(this->x + vec.x, this->y + vec.y);
  }

  // Overload the subtraction operator
  Vector2D operator-(const Vector2D& vec) const {
    return Vector2D(this->x - vec.x, this->y - vec.y);
  }

  // Overload the multiplication operator (vector * scalar)
  Vector2D operator*(double t) const {
    return Vector2D(this->x * t, this->y * t);
  }

  // Get the magnitude of the vector
  double magnitude() const { return sqrt(x * x + y * y); }

  // Get the unit vector of a vector
  Vector2D unit_vector() const { return *this * (1.0 / magnitude()); }
};

// Overload the standard output stream insertion operator
inline std::ostream& operator<<(std::ostream& out, const Vector2D& vec) {
  return out << vec.x << ' ' << vec.y;
}
}  // namespace graphics