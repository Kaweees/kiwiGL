#pragma once
#include "../include/vector2d.hpp"

namespace graphics {
class Triangle {
  public:
  // Constructor to initialize memory
  Triangle() = default;

  // Destructor to free the memory allocated
  ~Triangle() = default;

  Vector2D points[3];
};
}  // namespace graphics