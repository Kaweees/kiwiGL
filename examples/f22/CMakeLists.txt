add_executable(f22 f22.cpp)
target_link_libraries(f22 PRIVATE ${PROJECT_NAME})

# # Copy MNIST data files
# file(COPY ${PROJECT_SOURCE_DIR}/data/mnist DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)

# Setting our output directory
set_target_properties(f22 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
  OUTPUT_NAME f22
)
