#include <stddef.h>
#include <stdio.h>

#include <iostream>
#include <kiwigl/kiwigl.hpp>

//------------------------------------------------------------------------------------
// Program main entry point
//------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // Initialization of display
#ifndef BENCHMARK_MODE
  // Initialization of display
  kiwigl::Display* display =
      new kiwigl::Display("KiwiGL Stanford Bunny Demo", false, kiwigl::Vector3D(0, 0, -0.25),
                          kiwigl::Vector3D(M_PI, M_PI, 0), kiwigl::Vector3D(0.005, 0.005, 0.005));
#else
  kiwigl::Display display(10000);
#endif
  // Load the Stanford bunny mesh
  display->loadMesh("./assets/bunny.mesh");

  // Main graphics loop
  // Loop until window close button is pressed
  while (!display->shouldClose()) {
#ifdef BENCHMARK_MODE
    display->update();
#else
    display->processInput();
    display->update();
    display->render();
#endif
  }
  return 0;
}
