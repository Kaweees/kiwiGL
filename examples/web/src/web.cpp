#include <stddef.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <kiwigl/kiwigl.hpp>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

// The loop function that will be called by Emscripten
bool mainLoop(kiwigl::Display* display) {
  if (display) {
    display->processInput();
    display->update();
    if (display->shouldClose()) {
#ifdef __EMSCRIPTEN__
      emscripten_cancel_main_loop();
#endif
      return false;
    } else {
      display->render();
      return true;
    }
  } else {
    return false;
  }
}

// Wrapper function for emscripten_set_main_loop_arg
void runMainLoop(void* display) {
  mainLoop((kiwigl::Display*)display); // mainLoop handles calling emscripten_cancel_main_loop
}

//------------------------------------------------------------------------------------
// Program main entry point
//------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // Initialization of display
  kiwigl::Display* display =
      new kiwigl::Display("KiwiGL WebAssembly Demo", false, kiwigl::Vector3D(0, 0, -0.25),
                          kiwigl::Vector3D(M_PI, M_PI, 0), kiwigl::Vector3D(0.005, 0.005, 0.005));

  // Load the Stanford bunny mesh
  display->loadMesh("./assets/bunny.mesh");

#ifdef __EMSCRIPTEN__
  emscripten_set_main_loop_arg(runMainLoop, (void*)display, 0, true);
#else
  while (mainLoop(display)) { SDL_Delay(0); };
#endif
  delete display;
  display = nullptr;
  return 0;
}
