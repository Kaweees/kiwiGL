enable_testing()

set(
  TESTS
  test_operators
)

foreach (name IN LISTS TESTS)
  message(STATUS "Building test: ${name}")
  add_executable("${name}" "${name}.cpp")
  target_link_libraries(
    "${name}"
    PRIVATE
    ${PROJECT_NAME}
    GTest::gtest_main
  )
  set_target_properties("${name}" PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/" OUTPUT_NAME "${name}")
endforeach ()

