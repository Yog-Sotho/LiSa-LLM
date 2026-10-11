lisa_fetch_git(Catch2
    https://github.com/catchorg/Catch2.git
    v3.16.0
    317ac1ed4c0bb6e6b91eafc817e05c488feffcb3)

add_executable(lisa_tests
    test_config.cpp
    test_request.cpp
    test_utf8.cpp
    test_engine.cpp
)
target_link_libraries(lisa_tests PRIVATE lisa_core Catch2::Catch2WithMain)
lisa_configure_target(lisa_tests)

FetchContent_GetProperties(Catch2 SOURCE_DIR catch2_source_dir)
list(APPEND CMAKE_MODULE_PATH "${catch2_source_dir}/extras")
include(Catch)
# Engine tests read LISA_TEST_MODEL from the environment ctest runs in.
catch_discover_tests(lisa_tests DISCOVERY_MODE PRE_TEST)

# CLI smoke tests on the real binary.
add_test(NAME cli_version COMMAND lisa --version)
set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION "lisa ${PROJECT_VERSION}")
add_test(NAME cli_rejects_unknown_argument COMMAND lisa --bogus)
set_tests_properties(cli_rejects_unknown_argument PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_rejects_missing_config COMMAND lisa --config /nonexistent/config.yaml)
set_tests_properties(cli_rejects_missing_config PROPERTIES WILL_FAIL TRUE)
