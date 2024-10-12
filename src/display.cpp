#include "../include/display.hpp"

#include <SDL2/SDL.h>

#include "../include/vector3d.hpp"

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

  // Initialize the vertices
  // Start loading my array of vectors
  // From -1 to 1 (in this 9x9x9 cube)
  for (float x = -1; x <= 1; x += 0.25) {
    for (float y = -1; y <= 1; y += 0.25) {
      for (float z = -1; z <= 1; z += 0.25) {
        vertices.push_back(Vector3D(x, y, z));
      }
    }
  }

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
  // Rotate the vertices
  for (auto &vertex : vertices) {
    vertex.rotateX(0.01);
    vertex.rotateY(0.01);
    vertex.rotateZ(0.01);
  }
}

void Display::render() {
  // Clear the renderer
  clear();

  frameBuffer->drawFilledRectangle(64, 64, 128, 128, Color(0, 255, 0, 255));

  frameBuffer->drawGrid(Color(0xFF444444));

  for (auto &vertex : vertices) {
    Vector2D projectedPoint = vertex.project();
    // frameBuffer->drawPixel(projectedPoint.x + (frameBuffer->getWidth() / 2),
    //     projectedPoint.y + (frameBuffer->getHeight() / 2),
    //     Color(0xFFFFFF00));
    frameBuffer->drawFilledRectangle(
        projectedPoint.x + (frameBuffer->getWidth() / 2),
        projectedPoint.y + (frameBuffer->getHeight() / 2), 4, 4,
        Color(0xFFFFFF00));
  }

  // Update the texture with the frame buffer data
  SDL_UpdateTexture(texture, nullptr, frameBuffer->getData().data(),
      frameBuffer->getWidth() * sizeof(uint32_t));

  // Copy the texture to the renderer
  SDL_RenderCopy(renderer, texture, nullptr, nullptr);

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

  // Clear the frame buffer
  frameBuffer->clear(Color(0, 0, 0, 0));
}

bool Display::shouldClose() const { return SDL_QuitRequested(); }
}  // namespace graphics