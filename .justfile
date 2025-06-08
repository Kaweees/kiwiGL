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
run *package='f22':
  @./target/release/{{package}}

# Build the project
build *build_type='Release':
  @mkdir -p build
  @echo "Configuring the build system..."
  @cd build && cmake -S .. -B . -DCMAKE_BUILD_TYPE={{build_type}}
  @echo "Building the project..."
  @cd build && cmake --build . -j$(nproc)

# Remove build artifacts and non-essential files
clean:
  @echo "Cleaning..."
  @rm -rf build
  @rm -rf target

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
