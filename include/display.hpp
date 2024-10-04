#pragma once

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>

#include <memory>

#include "../include/frame_buffer.hpp"

namespace graphics {
class Display {
  private:
  // Constants for display
  SDL_Window *window;
  SDL_Texture *texture;
  SDL_Renderer *renderer;
  // std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window;
  // std::unique_ptr<SDL_Texture, decltype(&SDL_DestroyTexture)> texture;
  // std::unique_ptr<SDL_Surface, decltype(&SDL_FreeSurface)> surface;
  // std::unique_ptr<SDL_Event> event;
  // std::unique_ptr<SDL_Renderer> renderer;
  std::unique_ptr<FrameBuffer> frameBuffer;

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
