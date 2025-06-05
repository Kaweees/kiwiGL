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

<a href="https://github.com/Kaweees/kiwigl">
  <img alt="C++ Logo" src="assets/img/cpp.svg" align="right" width="150">
</a>

<div align="left">
  <h1><em><a href="https://github.com/Kaweees/kiwigl">~kiwigl</a></em></h1>
</div>

<!-- ABOUT THE PROJECT -->

A three-dimensional header-only graphics library written in C++13 and accelerated with CUDA/Apple Metal.

### Built With

[![C++][C++-shield]][C++-url]
[![CUDA][CUDA-shield]][CUDA-url]
[![Apple][Apple-shield]][Apple-url]
[![CMake][CMake-shield]][CMake-url]
[![Codecov][Codecov-shield]][Codecov-url]
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

Additionally, if you wish to utilize the GPU acceleration features, you will need to have [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) or [Apple Metal](https://developer.apple.com/metal/cpp/) installed on your machine.

### Installation

To get a local copy of the project up and running on your machine, follow these simple steps:

1. Clone the project repository

   ```sh
   git clone https://github.com/Kaweees/kiwigl.git
   cd kiwigl
   ```

2. Create a fresh build directory and navigate to it

   ```sh
   rm -rf build
   mkdir build
   cd build
   ```

3. Generate build files using CMake

   ```sh
   cmake -S .. -B . -DCMAKE_BUILD_TYPE=Debug
   ```

4. Build the entire project

   ```sh
   cmake --build .
   ```

### Building and Running Tests

1. Build only the tests

   ```sh
   cmake --build . --target tests
   ```

2. Run all tests

   ```sh
   ctest --output-on-failure
   ```

3. Run a specific test suite

   ```sh
   ./tests/test_suite_name
   ```

### Building and Running Examples

1. Build only the examples

   ```sh
   cmake --build . --target examples
   ```

2. Run a specific example

   ```sh
   ./examples/example_name
   ```

## Usage

<!--
### Benchmarks

Kiwigl is capable of rendering 3D scenes with thousands of triangles at interactive frame rates. The following benchmarks were conducted on a 2019 MacBook Pro with a 2.3 GHz 8-Core Intel Core i9 processor and 16 GB of RAM.

| Benchmark | Description | Result |
| --------- | ----------- | ------ |
| `cube` | Render a cube with 12 triangles | 60 FPS |
| `sphere` | Render a sphere with 960 triangles | 60 FPS |
| `bunny` | Render a Stanford Bunny with 69451 triangles | 60 FPS |
| `dragon` | Render a Stanford Dragon with 871306 triangles | 60 FPS |
-->

### Convention

Kiwigl uses the following conventions:

- [left-handed coordinate system](https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/coordinate-systems.html#:~:text=The%20differentiation%20between%20left%2Dhanded,a%20right%2Dhand%20coordinate%20system)
- [counter-clockwise winding order](https://learnwebgl.brown37.net/model_data/model_volume.html#:~:text=Winding%20Order%20for%20a%20triangle,the%20front%20of%20the%20triangle.) for triangle vertices

### Keyboard Controls

Kiwigl uses the following keyboard shortcuts:

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

The source code for kiwigl  is distributed under the terms of the GNU General Public License v3.0, as I firmly believe that collaborating on free and open-source software fosters innovations that mutually and equitably beneficial to both collaborators and users alike. See [`LICENSE`](./LICENSE) for details and more information.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/Kaweees/kiwigl.svg?style=for-the-badge
[contributors-url]: https://github.com/Kaweees/kiwigl/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Kaweees/kiwigl.svg?style=for-the-badge
[forks-url]: https://github.com/Kaweees/kiwigl/network/members
[stars-shield]: https://img.shields.io/github/stars/Kaweees/kiwigl.svg?style=for-the-badge
[stars-url]: https://github.com/Kaweees/kiwigl/stargazers

<!-- MARKDOWN SHIELD BAGDES & LINKS -->
<!-- https://github.com/Ileriayo/markdown-badges -->
[C++-shield]: https://img.shields.io/badge/C++-%23008080.svg?style=for-the-badge&logo=c%2B%2B&logoColor=004482&labelColor=222222&color=004482
[C++-url]: https://isocpp.org/
[CUDA-shield]: https://img.shields.io/badge/cuda-%23008080.svg?style=for-the-badge&logo=nVIDIA&logoColor=76B900&labelColor=222222&color=76B900
[CUDA-url]: https://developer.nvidia.com/cuda-zone
[Apple-shield]: https://img.shields.io/badge/metal-%23008080.svg?style=for-the-badge&logo=apple&logoColor=white&labelColor=222222&color=white
[Apple-url]: https://developer.apple.com/metal/
[CMake-shield]: https://img.shields.io/badge/CMake-%23008080.svg?style=for-the-badge&logo=cmake&logoColor=008FBA&labelColor=222222&color=008FBA
[CMake-url]: https://cmake.org/
[Codecov-shield]: https://img.shields.io/badge/codecov-%23008080.svg?style=for-the-badge&logo=codecov&logoColor=FF0077&labelColor=222222&color=FF0077
[Codecov-url]: https://codecov.io/
[github-actions-shield]: https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=2671E5&labelColor=222222&color=2671E5
[github-actions-url]: https://github.com/features/actions
