cmake_minimum_required(VERSION 3.14 FATAL_ERROR) # Set the minimum required
                                                 # version of CMake

project(
  kiwigl
  VERSION 1.0.0
  LANGUAGES CXX)

option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" OFF)
option(BENCHMARK "Build with benchmark mode" OFF)

# Library target
add_library(${PROJECT_NAME} INTERFACE)

# Set C++ standard
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CMAKE_CXX_STANDARD_REQUIRED ON
             CMAKE_CXX_STANDARD 17
             CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

target_include_directories(
  ${PROJECT_NAME}
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)

include(CheckLanguage)
include(FetchContent)

if(EMSCRIPTEN)
  if(APPLE)
    # Tell CMake we are *not* building for macOS and strip any inherited flags.
    set(CMAKE_OSX_ARCHITECTURES
        ""
        CACHE STRING "" FORCE)
    set(CMAKE_OSX_SYSROOT
        ""
        CACHE PATH "" FORCE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET
        ""
        CACHE STRING "" FORCE)
  endif()
  add_compile_definitions("__EMSCRIPTEN__")

  # WebAssembly specific settings
  target_link_options(
    ${PROJECT_NAME}
    INTERFACE
    "-sMODULARIZE=1"
    "-sEXPORT_NAME='Main'"
    "-sINVOKE_RUN=0"
    "-sDISABLE_DEPRECATED_FIND_EVENT_TARGET_BEHAVIOR=0"
    "-sEXPORTED_FUNCTIONS=['_main']"
    "-sEXPORTED_RUNTIME_METHODS=['callMain', 'ccall', 'cwrap']"
    "-sFORCE_FILESYSTEM=1"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sASYNCIFY"
    "-O2"
    "-sNO_EXIT_RUNTIME=1"
    "-sASSERTIONS=1")

  # Set output extension
  set(CMAKE_EXECUTABLE_SUFFIX ".js")
else()
  # Metal Support (macOS/iOS)
  if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)
    find_library(FOUNDATION_FRAMEWORK Foundation)

    if(METAL_FRAMEWORK AND METALKIT_FRAMEWORK)
      message(STATUS "Apple Metal found. Building with Metal support")
      enable_language(OBJCXX)
      set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES CMAKE_OBJCXX_STANDARD_REQUIRED ON
                   CMAKE_OBJCXX_STANDARD 20
                   MACOSX_BUNDLE ON)
      # Fetch Metal-cpp
      FetchContent_Declare(
        metal-cpp
        GIT_REPOSITORY https://github.com/rizajin/metal-cpp.git
        GIT_TAG "main"
        GIT_PROGRESS TRUE
        GIT_SHALLOW TRUE)
      FetchContent_MakeAvailable(metal-cpp)

      # Add metal-cpp include directories
      target_include_directories(${PROJECT_NAME}
                                 INTERFACE ${metal-cpp_SOURCE_DIR})

      target_link_libraries(
        ${PROJECT_NAME} INTERFACE "-framework Metal" "-framework Foundation"
                                  "-framework QuartzCore" "-framework MetalKit")
      add_compile_definitions("__METAL__")
    else()
      message(STATUS "Apple Metal not found. Building without Metal support")
    endif()
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
  else()
    if(WIN32)
      # Forcing MSVC to use utf-8 encoding
      add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
      add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    endif(WIN32)

    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      message(STATUS "CUDA found. Building with CUDA support")
      enable_language(CUDA)
      set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES CMAKE_CUDA_STANDARD 17
                   CMAKE_CUDA_STANDARD_REQUIRED ON
                   CMAKE_CUDA_SEPARABLE_COMPILATION ON)
      find_package(CUDA REQUIRED)
      include_directories(${CUDA_INCLUDE_DIRS})
      add_compile_definitions("__CUDA__")
    else()
      message(STATUS "CUDA not found. Building without CUDA support")
    endif()
  endif()
endif()

# Benchmark mode
if(BENCHMARK)
  target_compile_definitions(${PROJECT_NAME} PRIVATE BENCHMARK_MODE)
else()
  # Fetch SDL3
  FetchContent_Declare(
    SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG "release-3.2.16"
    GIT_PROGRESS TRUE
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(SDL3)

  # Add SDL3 include directories
  target_include_directories(${PROJECT_NAME}
                             INTERFACE ${SDL3_SOURCE_DIR}/include)

  # Link SDL3 libraries
  if(EMSCRIPTEN)
    target_link_libraries(${PROJECT_NAME} INTERFACE SDL3::SDL3-static)
  else()
    target_link_libraries(${PROJECT_NAME} INTERFACE SDL3::SDL3)
  endif()
endif()

# Examples
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Tests
if(NOT EMSCRIPTEN AND BUILD_TESTS)
  # Fetch GoogleTest
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG "v1.15.2"
    GIT_PROGRESS TRUE
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_subdirectory(tests)
endif()
