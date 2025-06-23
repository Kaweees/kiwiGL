#pragma once

#ifndef BENCHMARK_MODE
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#define SDL_QuitRequested()                                                                                            \
  (SDL_PumpEvents(), (SDL_PeepEvents(NULL, 0, SDL_PEEKEVENT, SDL_EVENT_QUIT, SDL_EVENT_QUIT) > 0))

#endif
#include <stdio.h>
#include <stdlib.h>

#include <memory>

#include "../core/constants.hpp"
#include "../geometry/mesh.hpp"
#include "../geometry/triangle.hpp"
#include "../geometry/vector.hpp"
#include "../graphics/frame_buffer.hpp"

#ifdef __CUDA__
#include "display.cuh"
#elif __METAL__
#include "display.mm"
#endif

namespace kiwigl {
// An enum to represent the various methods to render the display
enum RenderMethod {
  RENDER_WIRE,
  RENDER_WIRE_VERTEX,
  RENDER_FILL_TRIANGLE,
  RENDER_FILL_TRIANGLE_WIRE,
  RENDER_TEXTURED,
  RENDER_TEXTURED_WIRE
};

// Represents a display
class Display {
  private:
#ifndef BENCHMARK_MODE
    // Constants for display
    const SDL_DisplayMode* displayMode;
    bool fullScreen;
    SDL_Window* window;
    SDL_Texture* texture;
    SDL_Surface* surface;
    SDL_Renderer* renderer;
    SDL_Keycode key;
    uint64_t prevTime;
    RenderMethod renderMethod;
#else
    uint32_t count;
    uint32_t frameCount;
#endif
    std::unique_ptr<FrameBuffer> frameBuffer;
    Mesh mesh;
    std::vector<Triangle> projectedTriangles;

    Vector3D camera;
    Vector3D rotation;
    Vector3D rotationSpeed;
  public:
#ifndef BENCHMARK_MODE
    // Constructor to initialize memory
    Display(const std::string& title, bool fullscreen = false, const Vector3D& cameraPosition = Vector3D(0, 0, -5),
            const Vector3D& cameraRotation = Vector3D(0, 0, 0),
            const Vector3D& cameraRotationSpeed = Vector3D(0, 0, 0)) {
#else
    Display(uint32_t numOfFrames, const Vector3D& cameraPosition = Vector3D(0, 0, -5),
            const Vector3D& cameraRotation = Vector3D(0, 0, 0),
            const Vector3D& cameraRotationSpeed = Vector3D(0, 0, 0)) {
#endif
      // Initialize the camera
      camera = cameraPosition;
      rotation = cameraRotation;
      rotationSpeed = cameraRotationSpeed;
#ifndef BENCHMARK_MODE
      fullScreen = fullscreen;
      key = SDLK_UNKNOWN;
      prevTime = SDL_GetTicks();
      // Initialize SDL with only the required subsystems
      if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        fprintf(stderr, "SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        exit(EXIT_FAILURE);
      }

      // Query SDL for the display mode
      displayMode = SDL_GetCurrentDisplayMode(SDL_GetPrimaryDisplay());
      if (displayMode == nullptr) {
        fprintf(stderr, "SDL_GetCurrentDisplayMode failed: %s\n", SDL_GetError());
        exit(EXIT_FAILURE);
      }
#else
      count = 0;
      frameCount = numOfFrames;
#endif

      // Initialize the frame buffer
      frameBuffer = std::make_unique<FrameBuffer>(displayMode->w, displayMode->h);

#ifdef __CUDA__
      // Initialize the CUDA device pointers
      Face* d_faces = nullptr;
      Vector3D* d_vertices = nullptr;
      Triangle* d_projectedTriangles = nullptr;
      InitalizeCuda();
#elif __METAL__
      InitalizeMetal();
#endif

#ifndef BENCHMARK_MODE
#ifdef __EMSCRIPTEN__
      // Initialize the SDL window for Emscripten
      window = SDL_CreateWindow(title.c_str(), frameBuffer->getWidth(), frameBuffer->getHeight(), 0);
#else
      // Initialize the SDL window
      window = SDL_CreateWindow(title.c_str(), frameBuffer->getWidth(), frameBuffer->getHeight(),
                                fullScreen ? SDL_WINDOW_BORDERLESS : 0);
#endif
      if (window == nullptr) {
        fprintf(stderr, "Window could not be created! SDL_Error: %s\n", SDL_GetError());
        exit(EXIT_FAILURE);
      }

      // Initialize the SDL renderer
      renderer = SDL_CreateRenderer(window, nullptr);
      if (renderer == nullptr) {
        fprintf(stderr, "Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        exit(EXIT_FAILURE);
      }

#ifndef __EMSCRIPTEN__
      // Set the logical size of the renderer
      if (!SDL_SetRenderLogicalPresentation(renderer, frameBuffer->getWidth(), frameBuffer->getHeight(),
                                            SDL_LOGICAL_PRESENTATION_LETTERBOX)) {
        fprintf(stderr, "Failed to set logical size! SDL_Error: %s\n", SDL_GetError());
        exit(EXIT_FAILURE);
      }
#endif

      // Initialize the SDL texture
      texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
                                  frameBuffer->getWidth(), frameBuffer->getHeight());
      if (texture == nullptr) {
        fprintf(stderr, "Texture could not be created! SDL_Error: %s\n", SDL_GetError());
        exit(EXIT_FAILURE);
      }
#endif
    }

    // Destructor to free memory
    ~Display() {
#ifndef BENCHMARK_MODE
      // Free the texture
      if (texture != nullptr) {
        SDL_DestroyTexture(texture);
        texture = nullptr;
      }
      // Free the renderer
      if (renderer != nullptr) {
        SDL_DestroyRenderer(renderer);
        renderer = nullptr;
      }
      // Free the window
      if (window != nullptr) {
        SDL_DestroyWindow(window);
        window = nullptr;
      }
      // Quit SDL subsystems
      SDL_Quit();
#endif
    }

    // Method to process input
    void processInput() {
#ifndef BENCHMARK_MODE
      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        switch (event.type) {
          case SDL_EVENT_KEY_DOWN:
            key = event.key.key;
            // Reset rotation speeds when space is pressed
            if (key == SDLK_SPACE) { rotationSpeed = Vector3D(0, 0, 0); }
            break;
          default: key = SDLK_UNKNOWN;
        }
      }
#endif
    }

    // Method to update the display
    void update() {
#ifndef BENCHMARK_MODE
      while (SDL_GetTicks() < prevTime + FRAME_TIME);
      prevTime = SDL_GetTicks();
#endif

#ifdef __CUDA__
      LaunchCuda(frameBuffer->getWidth(), frameBuffer->getHeight());
#elif __METAL__
      LaunchMetal();
#else
  for (int i = 0; i < mesh.faces.size(); i++) {
    // Transform the vertices of the face
    Face face = mesh.faces[i];
    for (int j = 0; j < 3; j++) {
      // Transform the vertices
      Vector3D vertex = mesh.vertices[face.vertexIndices[j] - 1];

      // Rotate the vertices
      vertex.rotate(rotation.x, rotation.y, rotation.z);

      // Translate the vertices
      vertex.translate(camera.x, camera.y, -camera.z);

      // Scale the vertices
      vertex.scale(1.01, 1.01, 1.01);

      // Project the transformed vertices
      projectedTriangles[i].points[j] = vertex.project();

      // Translate the projected vertices to the center of the screen
      projectedTriangles[i].points[j].translate(frameBuffer->getWidth() / 2, frameBuffer->getHeight() / 2);
    }
  }
#endif
#ifndef BENCHMARK_MODE
      // Update rotation
      switch (key) {
        case SDLK_UP: rotationSpeed.x += 0.01; break;
        case SDLK_DOWN: rotationSpeed.x -= 0.01; break;
        case SDLK_LEFT: rotationSpeed.y += 0.01; break;
        case SDLK_RIGHT: rotationSpeed.y -= 0.01; break;
        default: break;
      }
      rotation.translate(rotationSpeed.x, rotationSpeed.y, rotationSpeed.z);
#endif
    }

    // Method to render the display
    void render() {
      // Clear the renderer
      clear();

      frameBuffer->drawGrid(Color(0xFF444444));

      for (auto& triangle : projectedTriangles) {
        for (int i = 0; i < 3; i++) {
          // Draw vertex points
          frameBuffer->drawFilledRectangle(triangle.points[i].x, triangle.points[i].y, 3, 3, Color(0xFFFFFF00));
        }
        // Draw triangle
        frameBuffer->drawTriangle(triangle.points[0].x, triangle.points[0].y, triangle.points[1].x,
                                  triangle.points[1].y, triangle.points[2].x, triangle.points[2].y, Color(0xFFFFFF00));
      }

#ifndef BENCHMARK_MODE
      // Update the texture with the frame buffer data
      SDL_UpdateTexture(texture, nullptr, frameBuffer->getData().data(), frameBuffer->getWidth() * sizeof(uint32_t));

      // Copy the texture to the renderer
      SDL_RenderTexture(renderer, texture, nullptr, nullptr);

      // Present the renderer
      SDL_RenderPresent(renderer);
#endif
    }

    // Method to clear the display
    void clear() {
#ifndef BENCHMARK_MODE
      // Clear the renderer
      SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
      SDL_RenderClear(renderer);
#endif

      // Clear the frame buffer
      frameBuffer->clear(Color(0, 0, 0, 0));
    }

    // Method to check if the display should close
    bool shouldClose() {
#ifndef BENCHMARK_MODE
      return SDL_QuitRequested();
#else
      count++;
      return count >= frameCount;
#endif
    }

#ifdef __CUDA__
    // Method to initialize CUDA
    virtual void InitalizeCuda();

    // Method to free CUDA
    virtual void FreeCuda();

    // Method to launch CUDA
    virtual void LaunchCuda(int width, int height);
#elif __METAL__
    // Method to initialize Metal
    virtual void InitalizeMetal();

    // Method to free Metal
    virtual void FreeMetal();

    // Method to launch Metal
    virtual void LaunchMetal();
#endif

    // Method to load a mesh
    void loadMesh(const std::string& filename) {
      mesh = Mesh();
      mesh.loadMesh(filename);
      projectedTriangles.resize(mesh.faces.size());
    }
};
} // namespace kiwigl
