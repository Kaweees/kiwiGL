# Like GNU `make`, but `just` rustier.
# https://just.systems/
# run `just` from this directory to see available commands

alias r := run
alias b := build
alias c := clean
alias t := test
alias f := format

# Default command when 'just' is run without arguments
default:
  @just --list

# Run a package
run *args='f22':
  ./target/release/{{args}}

# Build the project(Release or Debug)
build *args='Release':
  mkdir -p build
  echo "Configuring the build system..."
  cd build && cmake -S .. -B . -DCMAKE_BUILD_TYPE={{args}}
  echo "Building the project..."
  cd build && cmake --build .

# Clean the project
clean:
  echo "Cleaning build directory..."
  rm -rf build
  rm -rf target

# Run code quality tools
test:
  echo "Running tests..."
  ./target/release/test_libray

# Format the project
format:
  chmod +x ./scripts/format.sh
  ./scripts/format.sh format
  cmake-format -i CMakeLists.txt

# Generate documentation
docs:
  echo "Generating documentation..."
