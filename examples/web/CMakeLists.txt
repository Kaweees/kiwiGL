add_executable(web web.cpp)
target_link_libraries(web PRIVATE ${PROJECT_NAME})

# Setting our output directory
set_target_properties(web PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
  OUTPUT_NAME web
)
