#include <stddef.h>
#include <stdio.h>
#include <iostream>
#include <kiwigl/kiwigl.hpp>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

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

//------------------------------------------------------------------------------------
// Program main entry point
//------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // Initialization of display
  g_display = new kiwigl::Display();

  // Load the Stanford bunny mesh
  g_display->loadMesh("assets/bunny.obj");

#ifdef __EMSCRIPTEN__
  // Set the target FPS (60 in this case)
  emscripten_set_main_loop_timing(EM_TIMING_RAF, 60);

  // Set up the main loop for Emscripten
  emscripten_set_main_loop(mainLoop, 0, true);
#else
  // Traditional loop for native builds
  while (!g_display->shouldClose()) { mainLoop(); }
#endif

  // Cleanup
  delete g_display;
  return 0;
}
