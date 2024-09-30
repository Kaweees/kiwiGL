#pragma once

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>

namespace graphics {
class Display {
  // Constants for display
  SDL_Window* window;
  SDL_Renderer* renderer;

  public:
  // Constructor to initialize memory
  Display();

  // Destructor to free memory
  ~Display();

  // Method to clear the display
  void clear();

  // Method to update the display
  void update();

  // Method to check if the display should close
  bool shouldClose() const;
};
}  // namespace graphics