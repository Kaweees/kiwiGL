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

  void clear(const Color &color);

  int getWidth() const { return width; }
  int getHeight() const { return height; }

  // Method to draw a pixel on the display
  void drawPixel(int x, int y, const Color &color);

  // Method to draw a line on the display using Bresenham's line algorithm
  void drawLine(int x1, int y1, int x2, int y2, const Color &color);

  // 

  // Method to draw a grid on the display
  void drawGrid(const Color &color);

  // Method to draw a rectangle on the display
  void drawRectangle(int x, int y, int width, int height, const Color &color);

  // Method to draw a filled rectangle on the display
  void drawFilledRectangle(
      int x, int y, int width, int height, const Color &color);

  const std::vector<uint32_t> &getData() const { return buffer; }
};
}  // namespace graphics