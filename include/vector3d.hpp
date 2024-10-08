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

  // Overload the subtraction operator
  Vector3D operator-(const Vector3D& vec) const {
    return Vector3D(this->x - vec.x, this->y - vec.y, this->z - vec.z);
  }

  // Overload the addition operator
  Vector3D operator+=(const Vector3D& vec) {
    this->x += vec.x;
    this->y += vec.y;
    this->z += vec.z;
    return *this;
  }

  // Overload the multiplication operator (vector * scalar)
  Vector3D operator*=(double t) {
    this->x *= t;
    this->y *= t;
    this->z *= t;
    return *this;
  }

  // Overload the division operator (vector / scalar)
  Vector3D operator/=(double t) {
    this->x /= t;
    this->y /= t;
    this->z /= t;
    return *this;
  }

  // Get the magnitude of the vector
  double magnitude() const {
    return this->x * this->x + this->y * this->y + this->z * this->z;
  }

  // Get the length of the vector
  double length() const { return sqrt(magnitude()); }

  Vector2D project() const {
    return Vector2D(x * 128 / z, y * 128 / z);
  }
};

// Type alias for Point3D
using Point3D = Vector3D;

// Overload the standard output stream insertion operator
inline std::ostream& operator<<(std::ostream& out, const Vector3D& vec) {
  return out << vec.x << ' ' << vec.y << ' ' << vec.z;
}

// Overload the addition operator
inline Vector3D operator+(const Vector3D& u, const Vector3D& v) {
  return Vector3D(u.x + v.x, u.y + v.y, u.z + v.z);
}

// Overload the subtraction operator
inline Vector3D operator-(const Vector3D& u, const Vector3D& v) {
  return Vector3D(u.x - v.x, u.y - v.y, u.z - v.z);
}

// Overload the multiplication operator (vector * vector)
inline Vector3D operator*(const Vector3D& u, const Vector3D& v) {
  return Vector3D(u.x * v.x, u.y * v.y, u.z * v.z);
}

// Overload the multiplication operator (vector * scalar)
inline Vector3D operator*(double t, const Vector3D& vec) {
  return Vector3D(t * vec.x, t * vec.y, t * vec.z);
}

// Overload the multiplication operator (vector * scalar)
inline Vector3D operator*(const Vector3D& vec, double t) {
  return Vector3D(t * vec.x, t * vec.y, t * vec.z);
}

// Overload the division operator (vector / scalar)
inline Vector3D operator/(Vector3D vec, double t) {
  return Vector3D(vec.x / t, vec.y / t, vec.z / t);
}

// Get the dot product of two vectors
inline double dot(const Vector3D& u, const Vector3D& v) {
  return u.x * v.x + u.y * v.y + u.z * v.z;
}

// Get the cross product of two vectors
inline Vector3D cross(const Vector3D& u, const Vector3D& v) {
  return Vector3D(
      u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}

// Get the unit vector of a vector
inline Vector3D unit_vector(Vector3D vec) { return vec / vec.length(); }
}  // namespace graphics