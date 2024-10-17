<!--
*** This readme was adapted from Best-README-Template.
  https://github.com/othneildrew/Best-README-Template
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<div align="left">

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]

</div>

<a href="https://github.com/Kaweees/graphics">
  <img alt="C++ Logo" src="assets/img/cpp.svg" align="right" width="150">
</a>

<div align="left">
  <h1><em><a href="https://github.com/Kaweees/graphics">~graphics</a></em></h1>
</div>

<!-- ABOUT THE PROJECT -->

A three-dimensional graphics library from scratch written in C++13 and accelerated with CUDA/Apple Metal.

### Built With

[![Neovim][C++-shield]][C++-url]
[![CUDA][CUDA-shield]][CUDA-url]
[![Apple][Apple-shield]][Apple-url]
[![GitHub Actions][github-actions-shield]][github-actions-url]

<!-- PROJECT PREVIEW -->
## Preview

<p align="center">
  <img src="assets/img/demo.mp4"
  width = "80%"
  alt = "Video demonstration"
  />
</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Before attempting to build this project, make sure you have [Simple DirectMedia Layer (SDL 2)](https://wiki.libsdl.org/SDL2/Installation), [GNU Make](https://www.gnu.org/software/make/), and [CMake](https://cmake.org) installed on your machine.


### Installation

To get a local copy of the project up and running on your machine, follow these simple steps:

1. Clone the project repository

   ```sh
   git clone https://github.com/Kaweees/graphics.git
   cd graphics
   ```

2. Build and execute the project
   ```sh
   clear && cmake CMakeLists.txt && make && ./target/release/graphics
   ```

## Usage

### Convention

The graphics library uses the following conventions:

- right-handed coordinate system
- counter-clockwise winding order
- column-major matrices
- row-major vectors

- **Vectors**: `vec3` for 3D vectors, `vec4` for 4D vectors, and `vec2` for 2D vectors.
- **Matrices**: `mat3` for 3x3 matrices, `mat4` for 4x4 matrices.
- **Quaternions**: `quat` for quaternions.
- **Colors**: `color` for RGB colors.


### Keyboard Controls

To interact with the graphics library, use the following keyboard shortcuts:

| Command Keybind | Command Description |
| --------------- | ------------------- |
| <kbd>CTRL</kbd> + <kbd>q</kbd> | Quit the application |


<!-- PROJECT FILE STRUCTURE -->
## Project Structure

```sh
graphics/
├── .github/                       - GitHub Actions CI/CD workflows
├── include/                       - project header files
├── src/                           - project source files
│   └── main.c                     - Entry point, main function
├── CMakeLists.txt                 - CMake build script
├── LICENSE                        - project license
└── README.md                      - you are here
```

## License

The source code for my website is distributed under the terms of the GNU General Public License v3.0, as I firmly believe that collaborating on free and open-source software fosters innovations that mutually and equitably beneficial to both collaborators and users alike. See [`LICENSE`](./LICENSE) for details and more information.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/Kaweees/graphics.svg?style=for-the-badge
[contributors-url]: https://github.com/Kaweees/graphics/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Kaweees/graphics.svg?style=for-the-badge
[forks-url]: https://github.com/Kaweees/graphics/network/members
[stars-shield]: https://img.shields.io/github/stars/Kaweees/graphics.svg?style=for-the-badge
[stars-url]: https://github.com/Kaweees/graphics/stargazers

<!-- MARKDOWN SHIELD BAGDES & LINKS -->
<!-- https://github.com/Ileriayo/markdown-badges -->
[C++-shield]: https://img.shields.io/badge/C++-%23008080.svg?style=for-the-badge&logo=c%2B%2B&logoColor=004482&labelColor=222222&color=004482
[C++-url]: https://isocpp.org/
[CUDA-shield]: https://img.shields.io/badge/cuda-%23008080.svg?style=for-the-badge&logo=nVIDIA&logoColor=76B900&labelColor=222222&color=76B900
[CUDA-url]: https://developer.nvidia.com/cuda-zone
[Apple-shield]: https://img.shields.io/badge/metal-%23008080.svg?style=for-the-badge&logo=apple&logoColor=white&labelColor=222222&color=white
[Apple-url]: https://developer.apple.com/metal/
[github-actions-shield]: https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=2671E5&labelColor=222222&color=2671E5
[github-actions-url]: https://github.com/features/actions
