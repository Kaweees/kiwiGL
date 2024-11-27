#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "../graphics/color.hpp"

namespace kiwigl {

  // Represents the memory that holds the color values of the pixels
  class FrameBuffer {
    private:
    int width;
    int height;
    std::vector<uint32_t> buffer;

    public:
    FrameBuffer(int width, int height) : width(width), height(height), buffer(width * height, 0) {}
    ~FrameBuffer() = default;

    void clear(const Color &color) {
      uint32_t clearColor = static_cast<uint32_t>(color);
      std::fill(buffer.begin(), buffer.end(), clearColor);
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }

    // Method to draw a pixel on the display
    void drawPixel(int x, int y, const Color &color) {
      if (0 <= x && x < width && 0 <= y && y < height) {
        buffer[(width * y) + x] = static_cast<uint32_t>(color);
      }
    }

    // Method to draw a line on the display using Bresenham's line algorithm
    void drawLine(int x1, int y1, int x2, int y2, const Color &color) {
      // Find the difference in x and y coordinates
      int dx = std::abs(x2 - x1);
      int dy = std::abs(y2 - y1);

      // Find the sign of the difference
      int sx = (x1 < x2) ? 1 : -1;
      int sy = (y1 < y2) ? 1 : -1;

      // Initialize the error term
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

    // Method to draw a grid on the display
    void drawGrid(const Color &color) {
      for (int y = 0; y < height; y += 10) {
        for (int x = 0; x < width; x += 10) {
          drawPixel(x, y, color);
        }
      }
    }

    // Method to draw a rectangle on the display
    void drawRectangle(int x, int y, int width, int height, const Color &color) {
      // Draw the top line
      drawLine(x, y, x + width, y, color);

      // Draw the right line
      drawLine(x + width, y, x + width, y + height, color);

      // Draw the bottom line
      drawLine(x + width, y + height, x, y + height, color);

      // Draw the left line
      drawLine(x, y + height, x, y, color);
    }

    // Method to draw a filled rectangle on the display
    void drawFilledRectangle(int x, int y, int width, int height, const Color &color) {
      for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
          drawPixel(x + j, y + i, color);
        }
      }
    }

    // Method to draw a triangle on the display
    void drawTriangle(int x1, int y1, int x2, int y2, int x3, int y3, const Color &color) {
      drawLine(x1, y1, x2, y2, color);
      drawLine(x2, y2, x3, y3, color);
      drawLine(x3, y3, x1, y1, color);
    }

    const std::vector<uint32_t> &getData() const { return buffer; }
  };
}  // namespace kiwigl
