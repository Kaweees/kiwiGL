cmake_minimum_required(VERSION 3.14 FATAL_ERROR) # Set the minimum required version of CMake

project(kiwigl VERSION 1.0.0 LANGUAGES CXX)

option(BUILD_WASM "Build for WebAssembly" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" ON)
option(BENCHMARK "Build with benchmark mode" OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(BUILD_WASM)
  # Ensure we're using Emscripten compiler
  if(NOT DEFINED EMSCRIPTEN)
    message(FATAL_ERROR "BUILD_WASM requires using Emscripten CMake toolchain")
  endif()

  # WebAssembly specific settings
  set(CMAKE_EXECUTABLE_SUFFIX ".js")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_SDL=2 -s WASM=1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s ALLOW_MEMORY_GROWTH=1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap']")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --preload-file ${CMAKE_SOURCE_DIR}/public/assets@/assets")


  # Disable CUDA/Metal when building for web
  set(CUDA_FOUND FALSE)
  set(METAL_FOUND FALSE)
  set(BUILD_TESTS FALSE)
else()
  include(CheckLanguage)
  # Check for CUDA
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CMAKE_CUDA_STANDARD 14)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA)
    set(CUDA_FOUND TRUE)
    message(STATUS "CUDA found. Building with CUDA support.")
  else()
    set(CUDA_FOUND FALSE)
    message(STATUS "CUDA not found. Checking for Apple Metal support.")
  endif()

  # Check for Apple Metal
  if(NOT CUDA_FOUND AND APPLE)
    # Set the SDK path
    # set(CMAKE_OSX_SYSROOT /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk)
    # find_library(METAL_LIBRARY Metal)
    # find_library(FOUNDATION_LIBRARY Foundation)
    # if(METAL_LIBRARY AND FOUNDATION_LIBRARY)
    #   set(METAL_FOUND TRUE)
    #   target_compile_definitions(${PROJECT_NAME} PRIVATE USE_METAL)
    #   message(STATUS "Apple Metal found. Building with Metal support.")
    # else()
    #   message(WARNING "Neither CUDA nor Apple Metal found. Building without GPU acceleration.")
    # endif()
  endif()
endif()


# Library target
add_library(
  ${PROJECT_NAME}
  INTERFACE
)

target_include_directories(
  ${PROJECT_NAME}
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Benchmark mode
if(BENCHMARK)
    target_compile_definitions(${PROJECT_NAME} PRIVATE BENCHMARK_MODE)
else()
  # Find SDL2 package
  find_package(SDL2 REQUIRED)

  # Include SDL2 headers
  include_directories(${SDL2_INCLUDE_DIRS})

  # Link SDL2 libraries (corrected for INTERFACE library)
  target_link_libraries(${PROJECT_NAME} INTERFACE ${SDL2_LIBRARIES})
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    # Fetch GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG "v1.15.2"
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    add_subdirectory(tests)
endif()

