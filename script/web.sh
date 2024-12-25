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

# Create build directory
mkdir -p build
cd build

# Configure with Emscripten
emcmake cmake .. \
    -DBUILD_WASM=ON \
    -DBUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(nproc)

cd web

ls -la

# cp ../web/index.js ../target/release/

# # Copy web files
# cp ../web/index.html ../target/release/
