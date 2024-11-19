cmake_minimum_required(VERSION 3.8 FATAL_ERROR) # Set the minimum required version of CMake
project(kiwigl LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14) # Use C++14 standard for more features
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror") # Set the compiler flags to include all warnings and treat them as errors

# Declaring our executable
add_executable(${PROJECT_NAME})

# Set benchmark mode to true
# target_compile_definitions(${PROJECT_NAME} PRIVATE BENCHMARK_MODE)

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
  # Set the SDK path
  set(CMAKE_OSX_SYSROOT /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk)
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

# Adding our source files
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/kiwi/src/*.c" "${CMAKE_CURRENT_LIST_DIR}/kiwi/src/*.cpp" "${CMAKE_CURRENT_LIST_DIR}/kiwi/*.cpp" "${CMAKE_CURRENT_LIST_DIR}/kiwi/src/*.cxx" "${CMAKE_CURRENT_LIST_DIR}/kiwi/src/*.cu") # Define PROJECT_SOURCES as a list of all source files
target_sources(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCES})
# Adding our include files
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/kiwi/include)

# Setting ASSETS_PATH
target_compile_definitions(${PROJECT_NAME} PUBLIC ASSETS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/") # Set the asset path macro to the absolute path on the dev machine

# Setting our output directory
set_target_properties(${PROJECT_NAME} PROPERTIES 
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/target/release/"
  OUTPUT_NAME ${PROJECT_NAME}
)

# Find SDL2 package
find_package(SDL2 REQUIRED)

# Include SDL2 headers
include_directories(${SDL2_INCLUDE_DIRS})

# Link SDL2 libraries (add this to your target)
target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES})