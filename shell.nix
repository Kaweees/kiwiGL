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
  shellHook = "";
}
