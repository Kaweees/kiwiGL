#include <stddef.h>
#include <stdio.h>

#include <iostream>
#include <kiwigl/kiwigl.hpp>

#ifdef __METAL__
#include <cassert>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <simd/simd.h>

#include <AppKit/AppKit.hpp>
#include <Metal/Metal.hpp>
#include <MetalKit/MetalKit.hpp>
#endif

//------------------------------------------------------------------------------------
// Program main entry point
//------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // Initialization of display
#ifndef BENCHMARK_MODE
  kiwigl::Display display("KiwiGL F-22 Demo");
#else
  kiwigl::Display display(10000);
#endif
  // Load the F-22 mesh
  display.loadMesh("./assets/f22.mesh");

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
