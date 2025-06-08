# Like GNU `make`, but `just` rustier.
# https://just.systems/
# run `just` from this directory to see available commands

alias r := run
alias b := build
alias w := web
alias c := clean
alias t := test
alias f := format

# Default command when 'just' is run without arguments
default:
  @just --list

# Run a package
run *package='f22':
  @./target/release/{{package}}

# Build the project
build *build_type='Release':
  @mkdir -p build
  @echo "Configuring the build system..."
  @cd build && cmake -S .. -B . -DCMAKE_BUILD_TYPE={{build_type}}
  @echo "Building the project..."
  @cd build && cmake --build . -j$(nproc)

# Build the project for the web
web *build_type='Release':
  @mkdir -p build
  @echo "Configuring the build system..."
  @cd build && emcmake cmake -S .. -B . -DCMAKE_BUILD_TYPE={{build_type}} -DBUILD_WASM=1
  @echo "Building the project..."
  @cd build && cmake --build . -j$(nproc)
  @# Find and copy WASM, JS and data files to the public directory
  @find target/release/web -name "*.wasm" -exec cp {} ./public/ \;
  @find target/release/web -name "*.js" -exec cp {} ./public/ \;
  @find target/release/web -name "*.data" -exec cp {} ./public/ \;

# Remove build artifacts and non-essential files
clean:
  @echo "Cleaning..."
  @rm -rf build
  @rm -rf target
  @rm -rf .emscripten_cache

# Run code quality tools
test:
  @echo "Running tests..."
  @./target/release/kiwicpp_tests

# Format the project
format:
  @echo "Formatting..."
  @chmod +x ./scripts/format.sh
  @./scripts/format.sh format
  @cmake-format -i CMakeLists.txt

# Generate documentation
docs:
  @echo "Generating documentation..."
