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
      : vertexIndices{v1, v2, v3}, textures{t1, t2, t3}, color(color) {}

  // Destructor to free the memory allocated
  ~Face() = default;

  // The indices of the vertices
  int vertexIndices[3];

  // The indices of the textures
  Texture2D textures[3];

  // The color of the face
  Color color;
};
}  // namespace graphics
