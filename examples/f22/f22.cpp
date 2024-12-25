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

#ifdef __EMSCRIPTEN__
// Use emscripten_sleep only if EMTERPRETIFY is enabled
#ifdef EMTERPRETIFY_ASYNC
    emscripten_sleep(16); // ~60 FPS
#endif
#endif
  }
}

//------------------------------------------------------------------------------------
// Program main entry point
//------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  printf("Initializing display...\n");

  // Initialization of display
  g_display = new kiwigl::Display();

  // Load the mesh
  if (!g_display->loadMesh("assets/bunny.obj")) {
    printf("Failed to load mesh!\n");
    return 1;
  }

  printf("Starting main loop...\n");

#ifdef __EMSCRIPTEN__
  // Set up the main loop for Emscripten
  emscripten_set_main_loop(mainLoop, 0, 1);
#else
  // Traditional loop for native builds
  while (!g_display->shouldClose()) { mainLoop(); }
#endif

  // Cleanup
  delete g_display;
  return 0;
}
