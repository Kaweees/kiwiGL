set(EXECUTABLE_NAME web)

file(GLOB_RECURSE PROJECT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PROJECT_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

add_executable(${EXECUTABLE_NAME} ${PROJECT_SRCS})
target_include_directories(${EXECUTABLE_NAME} PUBLIC ${PROJECT_INCLUDES})
target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${PROJECT_NAME})

set(ASSETS_DIR ${CMAKE_SOURCE_DIR}/assets)
message(STATUS "ASSETS_DIR: ${ASSETS_DIR}")
target_link_options(${EXECUTABLE_NAME} PRIVATE "--preload-file"
                    "${ASSETS_DIR}@/assets")

# Setting our output directory
set_target_properties(
  ${EXECUTABLE_NAME}
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY
             "${PROJECT_SOURCE_DIR}/target/release/web/" OUTPUT_NAME
                                                         ${EXECUTABLE_NAME})
