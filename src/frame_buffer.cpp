#include "../include/frame_buffer.hpp"

namespace graphics {

FrameBuffer::FrameBuffer(int width, int height)
    : width(width), height(height), buffer(width * height, 0) {}

void FrameBuffer::setPixel(int x, int y, const Color& color) {
  if (x >= 0 && x < width && y >= 0 && y < height) {
    buffer[y * width + x] = static_cast<uint32_t>(color);
  }
}

void FrameBuffer::clear(const Color& color) {
  uint32_t clearColor = static_cast<uint32_t>(color);
  std::fill(buffer.begin(), buffer.end(), clearColor);
}
}  // namespace graphics