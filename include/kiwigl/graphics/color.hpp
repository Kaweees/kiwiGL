#pragma once

#include <stdint.h>

namespace kiwigl {

// Represents a color with red, green, blue, and alpha components
class Color {
  public:
    // Constructor to initialize memory
    Color()
      : red(0),
        green(0),
        blue(0),
        alpha(255) {}

    // Constructor to initialize memory with a 32-bit unsigned integer
    Color(uint32_t rgba)
      : red((rgba >> 0) & 0xFF),
        green((rgba >> 8) & 0xFF),
        blue((rgba >> 16) & 0xFF),
        alpha((rgba >> 24) & 0xFF) {}

    // Constructor to initialize memory with color values
    Color(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha)
      : red(red),
        green(green),
        blue(blue),
        alpha(alpha) {}

    // Destructor to free memory
    ~Color() = default;

    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t alpha;

    // Linearly interpolate between two colors
    Color lerp(const Color& other, float t) const {
      return Color(static_cast<uint8_t>(red + (other.red - red) * t),
                   static_cast<uint8_t>(green + (other.green - green) * t),
                   static_cast<uint8_t>(blue + (other.blue - blue) * t),
                   static_cast<uint8_t>(alpha + (other.alpha - alpha) * t));
    }

    // Cast the color as a 32-bit unsigned integer
    explicit operator uint32_t() const { return (alpha << 24) | (blue << 16) | (green << 8) | red; }
};
} // namespace kiwigl
