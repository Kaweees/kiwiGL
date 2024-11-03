#include "../include/color.hpp"

namespace graphics {
// Default to black if no color is specified
Color::Color() : red(0), green(0), blue(0), alpha(255) {}

// Initialize the color with the specified 32-bit unsigned integer
Color::Color(uint32_t rgba)
    : red((rgba >> 0) & 0xFF),
      green((rgba >> 8) & 0xFF),
      blue((rgba >> 16) & 0xFF),
      alpha((rgba >> 24) & 0xFF) {}

// Initialize the color with the specified values
Color::Color(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha)
    : red(red), green(green), blue(blue), alpha(alpha) {}

// Linearly interpolate between two colors
Color Color::lerp(const Color& other, float t) const {
  return Color(static_cast<uint8_t>(red + (other.red - red) * t),
      static_cast<uint8_t>(green + (other.green - green) * t),
      static_cast<uint8_t>(blue + (other.blue - blue) * t),
      static_cast<uint8_t>(alpha + (other.alpha - alpha) * t));
}

const Color Color::WHITE(255, 255, 255, 255);
}  // namespace graphics