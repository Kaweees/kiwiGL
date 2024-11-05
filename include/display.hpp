#pragma once

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>

#include <memory>

#include "../include/frame_buffer.hpp"
#include "../include/mesh.hpp"
#include "../include/triangle.hpp"
#include "../include/vector3d.hpp"

namespace graphics {
// An enum to represent the various methods to render the display
enum RenderMethod {
  RENDER_WIRE,
  RENDER_WIRE_VERTEX,
  RENDER_FILL_TRIANGLE,
  RENDER_FILL_TRIANGLE_WIRE,
  RENDER_TEXTURED,
  RENDER_TEXTURED_WIRE
};

// Represents a display
class Display {
  private:
  // Constants for display
  SDL_DisplayMode displayMode;
#ifndef BENCHMARK_MODE
  bool fullScreen;
  SDL_Window *window;
  SDL_Texture *texture;
  SDL_Surface *surface;
  SDL_Event *event;
  SDL_Renderer *renderer;
  SDL_Keycode keyPressed;
  uint32_t prevTime;
  RenderMethod renderMethod;
#else
  uint32_t count;
  uint32_t frameCount;
#endif
  std::unique_ptr<FrameBuffer> frameBuffer;
  Mesh mesh;
  std::vector<Triangle> projectedTriangles;
  Face *d_faces;
  Vector3D *d_vertices;
  Triangle *d_projectedTriangles;

  Vector3D camera;
  Vector3D rotation;
  Vector3D rotationSpeed;

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
  virtual void LaunchCuda(int width, int height);
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
