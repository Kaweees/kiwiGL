#pragma once
#include "../include/color.hpp"
#include "../include/texture.hpp"

namespace graphics {
// Represents a two-dimensional texture
class Face {
  public:
  // Constructor to initialize memory
  Face(int v1, int v2, int v3, const Texture2D& t1, const Texture2D& t2,
      const Texture2D& t3, const Color& color)
      : v1(v1), v2(v2), v3(v3), t1(t1), t2(t2), t3(t3), color(color) {}

  // Destructor to free the memory allocated
  ~Face() = default;

  // The first vertex index
  int v1;
  // The second vertex index
  int v2;
  // The third vertex index
  int v3;
  // The first texture
  Texture2D t1;
  // The second texture
  Texture2D t2;
  // The third texture
  Texture2D t3;
  // The color of the face
  Color color;
};
}  // namespace graphics
