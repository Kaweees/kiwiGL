#include "../include/display.hpp"

#include <SDL2/SDL.h>

#include "../include/vector.hpp"

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
  vertices = std::vector<Vector3D>{
      Vector3D(0, 0, 0),
      Vector3D(0, 0, 1),
      Vector3D(0, 1, 0),
      Vector3D(0, 1, 1),
      Vector3D(1, 0, 0),
      Vector3D(1, 0, 1),
      Vector3D(1, 1, 0),
      Vector3D(1, 1, 1),
  };

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

  frameBuffer->drawFilledRectangle(64, 64, 128, 128, Color(0, 255, 0, 255));

  frameBuffer->drawGrid(frameBuffer->getWidth(), frameBuffer->getHeight(), 32,
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
}  // namespace graphics