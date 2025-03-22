cmake_minimum_required(VERSION 3.14 FATAL_ERROR) # Set the minimum required version of CMake

project(kiwigl VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check for CUDA
include(CheckLanguage)
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

# Check for Apple Metal (only if CUDA is not found)
if(NOT CUDA_FOUND AND APPLE)
  set(CMAKE_THREAD_LIBS_INIT "-lpthread")
  set(CMAKE_HAVE_THREADS_LIBRARY 1)
  set(CMAKE_USE_WIN32_THREADS_INIT 0)
  set(CMAKE_USE_PTHREADS_INIT 1)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
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

# Optional builds
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" ON)
option(BENCHMARK "Build with benchmark mode" OFF)

# Benchmark mode
if(BENCHMARK)
  target_compile_definitions(${PROJECT_NAME} PRIVATE BENCHMARK_MODE)
else()
  # Fetch SDL2
  include(FetchContent)
  FetchContent_Declare(
    SDL2
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG "release-2.28.5"
    GIT_PROGRESS TRUE
  )
  FetchContent_MakeAvailable(SDL2)

  # Add SDL2 include directories
  target_include_directories(${PROJECT_NAME} INTERFACE
    ${SDL2_SOURCE_DIR}/include
  )

  # Link SDL2 libraries
  target_link_libraries(${PROJECT_NAME} INTERFACE SDL2::SDL2)
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
