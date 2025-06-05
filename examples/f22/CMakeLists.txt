set(EXECUTABLE_NAME f22)

add_executable(${EXECUTABLE_NAME} ${EXECUTABLE_NAME}.cpp)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${PROJECT_NAME} SDL2::SDL2)

# Setting our output directory
set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
  OUTPUT_NAME ${EXECUTABLE_NAME}
)
