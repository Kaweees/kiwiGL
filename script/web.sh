#!/bin/bash

# Format script called by the CI
# Usage:
#    format.sh format

#
#  Private Impl
#

# Ensure emscripten is available
if ! command -v emcmake &> /dev/null; then
    echo "emcmake not found. Please install and activate Emscripten"
    exit 1
fi

# Get CPU count based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    CPU_COUNT=$(sysctl -n hw.ncpu)
else
    # Linux and others
    CPU_COUNT=$(nproc)
fi

# Create build directory
mkdir -p build
cd build

# Configure with Emscripten
emcmake cmake .. \
    -DBUILD_WASM=ON \
    -DBUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release

# Check if configuration was successful
if [ $? -ne 0 ]; then
    echo "CMake configuration failed. Please check SDL2 installation."
    echo "You might need to install SDL2 for Emscripten with: emscripten/emsdk/upstream/emscripten/embuilder.py build sdl2"
    exit 1
fi

# Build
cmake --build . -j${CPU_COUNT}

# Check if build directory exists and create if needed
mkdir -p ../public

# Copy web files if they exist
if [ -d "src" ]; then
    # Find and copy WASM, JS and data files to the public directory
    find src -name "*.wasm" -exec cp {} ../public/ \;
    find src -name "*.js" -exec cp {} ../public/ \;
    find src -name "*.data" -exec cp {} ../public/ \;
else
    # Alternative locations to search for output files
    find . -name "*.wasm" -exec cp {} ../public/ \;
    find . -name "*.js" -exec cp {} ../public/ \;
    find . -name "*.data" -exec cp {} ../public/ \;
fi

cd ..
echo "Build complete. Files in public directory:"
ls -la public/
