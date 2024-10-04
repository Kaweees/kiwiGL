#pragma once
#include <cstdint>

namespace graphics {

// Represents a color with red, green, blue, and alpha components
class Color {
  public:
  // Constructor to initialize memory
  Color();

  // Constructor to initialize memory with a 32-bit unsigned integer
  Color(uint32_t rgba);

  // Constructor to initialize memory with color values
  Color(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha);

  // Destructor to free memory
  ~Color() = default;

  uint8_t red;
  uint8_t green;
  uint8_t blue;
  uint8_t alpha;

  // Cast the color as a 32-bit unsigned integer
  explicit operator uint32_t() const {
    return (alpha << 24) | (blue << 16) | (green << 8) | red;
  }
};
}  // namespace graphics