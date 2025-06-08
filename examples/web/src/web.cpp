#include <stddef.h>
#include <stdio.h>
#include <iostream>
#include <kiwigl/kiwigl.hpp>

#include <emscripten.h>
#include <emscripten/html5.h>

// Global display object since the loop function needs access
kiwigl::Display* g_display = nullptr;

// The loop function that will be called by Emscripten
void mainLoop() {
  if (g_display) {
    g_display->processInput();
    g_display->update();
    g_display->render();
  }
}

// Cleanup function to be called when the application exits
void cleanup() {
  if (g_display) {
    delete g_display;
    g_display = nullptr;
  }
}

//------------------------------------------------------------------------------------
// Program main entry point
//------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // Initialization of display
  g_display = new kiwigl::Display();

  // Load the Stanford bunny mesh
  g_display->loadMesh("assets/bunny.obj");

  // Register cleanup function to be called on exit
  atexit(cleanup);

  // Set the timing mode before setting up the main loop
  emscripten_set_main_loop_timing(EM_TIMING_RAF, 1);

  // Set up the main loop for Emscripten
  emscripten_set_main_loop(mainLoop, 0, true);

  return 0;
}
