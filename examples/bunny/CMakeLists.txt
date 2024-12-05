add_executable(bunny bunny.cpp)
target_link_libraries(bunny PRIVATE ${PROJECT_NAME})

# Setting our output directory
set_target_properties(bunny PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
  OUTPUT_NAME bunny
)
