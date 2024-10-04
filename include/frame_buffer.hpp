#pragma once

#include <cstdint>
#include <vector>

#include "color.hpp"

namespace graphics {

// Represents the memory that holds the color values of the pixels
class FrameBuffer {
  private:
  int width;
  int height;
  std::vector<uint32_t> buffer;

  public:
  FrameBuffer(int width, int height);
  ~FrameBuffer() = default;

  void setPixel(int x, int y, const Color& color);
  void clear(const Color& color);

  int getWidth() const { return width; }
  int getHeight() const { return height; }
  const std::vector<uint32_t>& getData() const { return buffer; }
};
}  // namespace graphics