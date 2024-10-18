#include "../include/display.hpp"

#include "../include/constants.hpp"

#ifdef USE_CUDA
#include "../include/display.cuh"
#elif USE_METAL
#include "../include/display.metal"
#endif

#include <SDL2/SDL.h>

#include "../include/constants.hpp"
#include "../include/vector3d.hpp"

namespace graphics {
Display::Display() {
  // Initialize the camera
  camera = Vector3D(0, 0, -5);
  rotation = Vector3D(0, 0, 0);

  prevTime = SDL_GetTicks();

  // Initialize SDL
  if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
    fprintf(
        stderr, "SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
    exit(EXIT_FAILURE);
  }

  // Query SDL for the display mode
  if (SDL_GetCurrentDisplayMode(0, &displayMode) != 0) {
    fprintf(stderr, "SDL_GetCurrentDisplayMode failed: %s\n", SDL_GetError());
    exit(EXIT_FAILURE);
  }

  // Initialize the frame buffer
  frameBuffer = std::make_unique<FrameBuffer>(displayMode.w, displayMode.h);

#ifdef USE_CUDA
  InitalizeCuda();
#elif USE_METAL
  // InitalizeMetal();
#else
  vertices.resize(NUM_VERTICES);
  projectedVertices.resize(NUM_VERTICES);
#endif
  int numVertices = 0;
  // Initialize the vertices
  // Start loading my array of vectors
  // From -1 to 1 (in this 9x9x9 cube)
  for (float x = -1; x <= 1; x += 0.25) {
    for (float y = -1; y <= 1; y += 0.25) {
      for (float z = -1; z <= 1; z += 0.25) {
        if (numVertices >= NUM_VERTICES) {
          break;
        }
        vertices[numVertices] = Vector3D(x, y, z);
        numVertices++;
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
  int currentTime = SDL_GetTicks();
  int deltaTime = currentTime - prevTime;
  prevTime = currentTime;

  if (true) {
#ifdef USE_CUDA
    LaunchCuda();
#elif USE_METAL
    LaunchMetal();
#else
    for (int i = 0; i < vertices.size(); i++) {
      // Transform the vertices
      auto vertex = vertices[i];

      // Rotate the vertices
      vertex.rotate(rotation.x, rotation.y, rotation.z);

      // Translate the vertices
      vertex.translate(camera.x, camera.y, -camera.z);

      // Scale the vertices
      // vertex.scale(1.01, 1.01, 1.01);

      // Project the transformed vertices
      projectedVertices[i] = vertex.project();
    }
#endif
    // Update rotation
    rotation.translate(0.01, 0, 0);
    deltaTime = 0;
  }
}

void Display::render() {
  // Clear the renderer
  clear();

  // frameBuffer->drawFilledRectangle(64, 64, 128, 128, Color(0, 255, 0, 255));

  // frameBuffer->drawGrid(Color(0xFF444444));

  for (auto &vertex : projectedVertices) {
    frameBuffer->drawFilledRectangle(vertex.x + (frameBuffer->getWidth() / 2),
        vertex.y + (frameBuffer->getHeight() / 2), 4, 4, Color(0xFFFFFF00));
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