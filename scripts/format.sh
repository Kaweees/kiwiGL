#!/bin/bash

# Format script called by the CI
# Usage:
#    format.sh format

#
#  Private Impl
#

format() {
  # Find all C/C++ files
  files=$(find . -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.cc" -o -name "*.hh" -o -name "*.cxx" -o -name "*.hxx" -o -name "*.cu" -o -name "*.cuh")

  if [ -z "$files" ]; then
    echo "No C/C++ files found"
    return 0
  fi

  # Format each file and capture any errors
  error=0
  for file in $files; do
    if ! clang-format -style=file -i "$file"; then
      echo "Error formatting $file"
      error=1
    fi
  done

  if [ $error -eq 0 ]; then
    echo "Successfully formatted all files"
    return 0
  else
    echo "Errors occurred during formatting"
    return 1
  fi
}

# Main script logic
set -e # Exit on error
case "$1" in
  format)
    format
    ;;
  *)
    echo "Usage: $0 {format}"
    exit 1
    ;;
esac
