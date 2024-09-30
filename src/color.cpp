#include "../include/color.hpp"

namespace graphics {
// Default to black if no color is specified
Color::Color() : red(0), green(0), blue(0), alpha(255) {}

// Initialize the color with the specified values
Color::Color(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha)
    : red(red), green(green), blue(blue), alpha(alpha) {}

// Color::Color(uint32_t rgba)
//     : r((rgba >> 0) & 0xFF),
//       g((rgba >> 8) & 0xFF),
//       b((rgba >> 16) & 0xFF),
//       a((rgba >> 24) & 0xFF) {}

// ... other method implementations ...

// Color Color::lerp(const Color& other, float t) const {
//   return Color(static_cast<uint8_t>(r + (other.r - r) * t),
//       static_cast<uint8_t>(g + (other.g - g) * t),
//       static_cast<uint8_t>(b + (other.b - b) * t),
//       static_cast<uint8_t>(a + (other.a - a) * t));
// }

}  // namespace graphics