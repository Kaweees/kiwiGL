#pragma once

namespace kiwigl {
// Represents a two-dimensional texture
class Texture2D {
  public:
    // Constructor to initialize memory
    Texture2D()
      : u(0),
        v(0) {}

    Texture2D(double u, double v)
      : u(u),
        v(v) {}

    // Destructor to free the memory allocated
    ~Texture2D() = default;

    // The horizontal coordinate of the texture
    double u;
    // The vertical coordinate of the texture
    double v;
};
} // namespace kiwigl
