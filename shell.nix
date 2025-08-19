{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    clang # C++ compiler
    cppcheck # C++ linter
    cmake # CMake build system
    cmake-format # CMake format tool
    nixfmt # Nix formatter
    just # Just runner
    pkg-config # Package configuration
    emscripten # Emscripten for web assembly
    nodejs # Required for Emscripten
    ninja # Ninja build system
    git # Git for version control
  ];

  # Shell hook to set up environment
  shellHook = ''
    # Set Emscripten cache to a writable directory within the project
    export EM_CACHE_DIR="$PWD/.emscripten_cache"
    export EM_CACHE="$EM_CACHE_DIR"
    export EMCC_TEMP_DIR="$EM_CACHE_DIR/tmp"
  '';
}
