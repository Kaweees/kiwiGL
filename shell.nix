{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    cmake # CMake build system
    cmake-format # CMake format tool
    clang # C++ compiler
    just # Just runner
    pkg-config # Package configuration
    emscripten # Emscripten for web assembly
    nodejs # Required for Emscripten
  ];

  # Shell hook to set up environment
  shellHook = ''
    # Set Emscripten cache to a writable directory within the project
    export EM_CACHE_DIR="$PWD/.emscripten_cache"
    mkdir -p "$EM_CACHE_DIR"
    export EM_CACHE="$EM_CACHE_DIR"
    export EMCC_TEMP_DIR="$EM_CACHE_DIR/tmp"
    mkdir -p "$EMCC_TEMP_DIR"
    echo "Emscripten cache (EM_CACHE) set to: $EM_CACHE"
    echo "Emscripten temp dir (EMCC_TEMP_DIR) set to: $EMCC_TEMP_DIR"
  '';
}
