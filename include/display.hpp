#pragma once

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>

#include <memory>

#include "../include/frame_buffer.hpp"
#include "../include/vector3d.hpp"

namespace graphics {
class Display {
  private:
  // Constants for display
  SDL_Window *window;
  SDL_Texture *texture;
  SDL_Surface *surface;
  SDL_Event *event;
  SDL_Renderer *renderer;
  std::unique_ptr<FrameBuffer> frameBuffer;
  std::vector<Vector3D> vertices;
  std::vector<Vector2D> projectedVertices;

  Vector3D camera;
  Vector3D rotation;

  int prevTime;

  public:
  // Constructor to initialize memory
  Display();

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
  bool shouldClose() const;
};
}  // namespace graphics
