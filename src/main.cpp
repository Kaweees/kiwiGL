#include <stddef.h>
#include <stdio.h>

#include "../include/display.hpp"

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

//------------------------------------------------------------------------------------
// Program main entry point
//------------------------------------------------------------------------------------
int main(int argc, char **argv) {
  // Initialization of Chip8 and loading of program

  graphics::Display display;

  // Main emulation loop
  // Loop until window close button or ESC key is pressed
  while (!display.shouldClose()) {
    display.processInput();
    display.update();
    display.render();
  }
  return EXIT_SUCCESS;
}