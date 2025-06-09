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
  kiwigl::Display display("KiwiGL Cube Demo");
#else
  kiwigl::Display display(10000);
#endif
  // Load the Cube mesh
  display.loadMesh("./assets/cube.mesh");

  // Main graphics loop
  // Loop until window close button is pressed
  while (!display.shouldClose()) {
#ifdef BENCHMARK_MODE
    display.update();
#else
    display.processInput();
    display.update();
    display.render();
#endif
  }
  return 0;
}
