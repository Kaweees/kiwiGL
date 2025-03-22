{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  buildInputs = with pkgs; [ cmake clang SDL2 pkg-config ];

  # Simple shell hook for C++14
  shellHook = ''
    export CMAKE_CXX_FLAGS="-std=c++14"
  '';
}
