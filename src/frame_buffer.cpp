#include "../include/frame_buffer.hpp"

namespace graphics {

FrameBuffer::FrameBuffer(int width, int height)
    : width(width), height(height), buffer(width * height, 0) {}

void FrameBuffer::drawPixel(int x, int y, const Color &color) {
  if (x >= 0 && x < width && y >= 0 && y < height) {
    buffer[y * width + x] = static_cast<uint32_t>(color);
  }
}

void FrameBuffer::drawLine(int x1, int y1, int x2, int y2, const Color &color) {
  // Bresenham's line algorithm
  int dx = abs(x2 - x1);
  int dy = abs(y2 - y1);
  int sx = (x1 < x2) ? 1 : -1;
  int sy = (y1 < y2) ? 1 : -1;
  int err = dx - dy;

  while (true) {
    drawPixel(x1, y1, color);

    if (x1 == x2 && y1 == y2) {
      break;
    }

    int e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x1 += sx;
    }

    if (e2 < dx) {
      err += dx;
      y1 += sy;
    }
  }
}

void FrameBuffer::drawGrid(
    int width, int height, int cellSize, const Color &color) {
  // Draw the vertical lines
  for (int x = 0; x <= width; x += cellSize) {
    drawLine(x, 0, x, height, color);
  }

  // Draw the horizontal lines
  for (int y = 0; y <= height; y += cellSize) {
    drawLine(0, y, width, y, color);
  }
}

void FrameBuffer::drawRectangle(
    int x, int y, int width, int height, const Color &color) {
  // Draw the top line
  drawLine(x, y, x + width, y, color);

  // Draw the right line
  drawLine(x + width, y, x + width, y + height, color);

  // Draw the bottom line
  drawLine(x + width, y + height, x, y + height, color);

  // Draw the left line
  drawLine(x, y + height, x, y, color);
}

void FrameBuffer::drawFilledRectangle(
    int x, int y, int width, int height, const Color &color) {
  for (int i = 0; i < height; i++) {
    drawLine(x, y + i, x + width, y + i, color);
  }
}

void FrameBuffer::clear(const Color &color) {
  uint32_t clearColor = static_cast<uint32_t>(color);
  std::fill(buffer.begin(), buffer.end(), clearColor);
}
}  // namespace graphics