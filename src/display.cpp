#include "../include/display.hpp"

#include <SDL2/SDL.h>

namespace graphics {
Display::Display() {
  // Initialize SDL
  if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
    fprintf(
        stderr, "SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
    exit(EXIT_FAILURE);
  }

  // Query SDL for the display mode
  SDL_DisplayMode displayMode;
  if (SDL_GetCurrentDisplayMode(0, &displayMode) != 0) {
    fprintf(stderr, "SDL_GetCurrentDisplayMode failed: %s\n", SDL_GetError());
    exit(EXIT_FAILURE);
  }

  // Initialize the frame buffer
  frameBuffer = std::make_unique<FrameBuffer>(displayMode.w, displayMode.h);

  // Initialize the SDL window
  window = SDL_CreateWindow("SDL Tutorial", SDL_WINDOWPOS_CENTERED,
      SDL_WINDOWPOS_CENTERED, frameBuffer->getWidth(), frameBuffer->getHeight(),
      SDL_WINDOW_BORDERLESS);
  if (window == nullptr) {
    fprintf(
        stderr, "Window could not be created! SDL_Error: %s\n", SDL_GetError());
    exit(EXIT_FAILURE);
  }

  // Initialize the SDL renderer
  renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  if (renderer == nullptr) {
    fprintf(stderr, "Renderer could not be created! SDL_Error: %s\n",
        SDL_GetError());
    exit(EXIT_FAILURE);
  }

  // Initialize the SDL texture
  texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
      SDL_TEXTUREACCESS_STREAMING, frameBuffer->getWidth(),
      frameBuffer->getHeight());
  if (texture == nullptr) {
    fprintf(stderr, "Texture could not be created! SDL_Error: %s\n",
        SDL_GetError());
    exit(EXIT_FAILURE);
  }

  // Set the blend mode
  SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);
}

Display::~Display() {
  // Free the frame buffer
  frameBuffer.reset();

  // Free the window
  SDL_DestroyWindow(window);
  window = nullptr;

  // Free the renderer
  SDL_DestroyRenderer(renderer);

  // Free the texture
  SDL_DestroyTexture(texture);
  texture = nullptr;

  // Quit SDL subsystems
  SDL_Quit();
}

void Display::update() {
  // Update the renderer
}

void Display::render() {
  // Clear the renderer
  clear();

  drawFilledRectangle(64, 64, 128, 128, Color(0, 255, 0, 255));

  drawGrid(frameBuffer->getWidth(), frameBuffer->getHeight(), 32,
      Color(255, 0, 0, 255));

  // Update the texture with the frame buffer data
  SDL_UpdateTexture(texture, nullptr, frameBuffer->getData().data(),
      frameBuffer->getWidth() * sizeof(uint32_t));

  // Copy the texture to the renderer
  SDL_RenderCopy(renderer, texture, nullptr, nullptr);

  // Clear the frame buffer
  frameBuffer->clear(Color(0, 255, 255, 255));

  // Present the renderer
  SDL_RenderPresent(renderer);
}

void Display::processInput() {
  SDL_Event event;
  switch (event.type) {
    default:
      break;
  }
}

void Display::clear() {
  // Clear the renderer
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
  SDL_RenderClear(renderer);
}

bool Display::shouldClose() const { return SDL_QuitRequested(); }

void Display::drawPixel(int x, int y, const Color &color) {
  frameBuffer->setPixel(x, y, color);
}

void Display::drawLine(int x1, int y1, int x2, int y2, const Color &color) {
  // Bresenham's line algorithm
  int dx = abs(x2 - x1);
  int dy = abs(y2 - y1);
  int sx = (x1 < x2) ? 1 : -1;
  int sy = (y1 < y2) ? 1 : -1;
  int err = dx - dy;

  while (true) {
    frameBuffer->setPixel(x1, y1, color);

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

void Display::drawGrid(
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

void Display::drawRectangle(
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

void Display::drawFilledRectangle(
    int x, int y, int width, int height, const Color &color) {
  for (int i = 0; i < height; i++) {
    drawLine(x, y + i, x + width, y + i, color);
  }
}
}  // namespace graphics