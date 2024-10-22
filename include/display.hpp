#pragma once

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>

#include <memory>

#include "../include/frame_buffer.hpp"
#include "../include/mesh.hpp"  // Add this line
#include "../include/vector3d.hpp"

namespace graphics {
class Display {
  private:
  // Constants for display
  SDL_DisplayMode displayMode;
#ifndef BENCHMARK_MODE
  SDL_Window *window;
  SDL_Texture *texture;
  SDL_Surface *surface;
  SDL_Event *event;
  SDL_Renderer *renderer;
  SDL_Keycode keyPressed;
  uint32_t prevTime;
#else
  uint32_t count;
  uint32_t frameCount;
#endif
  std::unique_ptr<FrameBuffer> frameBuffer;
  std::vector<Vector3D> vertices;
  std::vector<Vector2D> projectedVertices;
  Vector3D *d_vertices;
  Vector2D *d_projectedVertices;

  Vector3D camera;
  Vector3D rotation;
  Vector3D rotationSpeed;

  Mesh mesh;  // Add a mesh member

  public:
#ifndef BENCHMARK_MODE
  // Constructor to initialize memory
  Display();
#else
  Display(uint32_t numOfFrames);
#endif

  // Destructor to free memory
  ~Display();

  // Method to process input
  void processInput();

  // Method to update the display
  void update();

  // Method to render the display
  void render();

  // Method to clear the display
  void clear();

  // Method to check if the display should close
  bool shouldClose();

#ifdef USE_CUDA
  // Method to initialize CUDA
  virtual void InitalizeCuda();

  // Method to free CUDA
  virtual void FreeCuda();

  // Method to launch CUDA
  virtual void LaunchCuda();
#elif USE_METAL
  // Method to initialize Metal
  virtual void InitalizeMetal();

  // Method to free Metal
  virtual void FreeMetal();

  // Method to launch Metal
  virtual void LaunchMetal();
#endif
};
}  // namespace graphics
