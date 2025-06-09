# Like GNU `make`, but `just` rustier.
# https://just.systems/
# run `just` from this directory to see available commands

alias r := run
alias b := build
alias w := web
alias c := clean
alias ch := check
alias d := docs
alias t := test
alias f := format

# Default command when 'just' is run without arguments
default:
  @just --list

# Run a package
run *package='bunny':
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
  @if [ -n "${EM_CACHE-}" ]; then mkdir -p "$EM_CACHE" && mkdir -p "$EM_CACHE/tmp"; fi
  @echo "Configuring the build system..."
  @cd build && emcmake cmake -S .. -B . -DCMAKE_BUILD_TYPE={{build_type}}
  @echo "Building the project..."
  @cd build && cmake --build . -j$(nproc)
  @mkdir -p public/assets/
  @# Find and copy WASM, JS and data files to the public directory
  @find target/release/web -name "*.wasm" -exec cp {} ./public/ \;
  @find target/release/web -name "*.js" -exec cp {} ./public/ \;
  @find target/release/web -name "*.data" -exec cp {} ./public/ \;
  @find assets/ -name "*.obj" -exec cp {} ./public/assets/ \;
  @find assets/ -name "*.mesh" -exec cp {} ./public/assets/ \;

# Remove build artifacts and non-essential files
clean:
  @echo "Cleaning..."
  @rm -rf build
  @rm -rf target
  @if [ -n "${EM_CACHE-}" ]; then rm -rf "$EM_CACHE"; fi

# Run code quality tools
check:
  @echo "Running code quality tools..."
  @cppcheck --enable=all --suppress=missingInclude --suppress=unusedFunction --error-exitcode=1 include/kiwigl

# Run code quality tools
test:
  @echo "Running tests..."
  @./target/release/tests/test_library

# Format the project
format:
  @echo "Formatting..."
  @chmod +x ./scripts/format.sh
  @./scripts/format.sh format
  @cmake-format -i CMakeLists.txt

# Generate documentation
docs:
  @echo "Generating documentation..."
  rm -rf website
  ./docs/doxygen/m.css/documentation/doxygen.py --debug docs/doxygen/conf.py
  mkdir website
  cp -r docs/doxygen/html/* website
  cp -r docs/images website
  rm -rf docs/doxygen/latex docs/doxygen/xml docs/doxygen/html docs/doxygen/__pycache__
