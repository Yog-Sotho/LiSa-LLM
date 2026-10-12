lisa_fetch_git(Catch2
    https://github.com/catchorg/Catch2.git
    v3.16.0
    317ac1ed4c0bb6e6b91eafc817e05c488feffcb3)

add_executable(lisa_tests
    main.cpp
    sandbox_probes.cpp
    test_config.cpp
    test_request.cpp
    test_utf8.cpp
    test_stop.cpp
    test_openai.cpp
    test_engine.cpp
    test_server.cpp
    test_ipc.cpp
    test_sandbox.cpp
    test_http_limits.cpp
    test_auth.cpp
    test_admission.cpp
)
if(LISA_TLS)
    target_sources(lisa_tests PRIVATE test_tls.cpp)
    target_link_libraries(lisa_tests PRIVATE OpenSSL::SSL OpenSSL::Crypto)
endif()
# A worker that violates the protocol, for SandboxedEngine's robustness tests.
add_executable(hostile_worker hostile_worker.cpp)
target_link_libraries(hostile_worker PRIVATE lisa_core)
lisa_configure_target(hostile_worker)
target_compile_definitions(lisa_tests PRIVATE LISA_HOSTILE_WORKER="$<TARGET_FILE:hostile_worker>")
add_dependencies(lisa_tests hostile_worker)

# Custom main: the test binary also acts as the sandboxed worker.
target_link_libraries(lisa_tests PRIVATE lisa_core httplib::httplib Catch2::Catch2)
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
add_test(NAME cli_rejects_bad_port COMMAND lisa --port http)
set_tests_properties(cli_rejects_bad_port PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_hash_key COMMAND sh -c "printf 'abc\\n' | \"$0\" --hash-key" $<TARGET_FILE:lisa>)
set_tests_properties(cli_hash_key PROPERTIES PASS_REGULAR_EXPRESSION
    "^sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n$")
# --check-config only needs model_path to be a readable regular file.
add_test(NAME cli_check_config COMMAND lisa --check-config --model ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
set_tests_properties(cli_check_config PROPERTIES PASS_REGULAR_EXPRESSION "configuration OK")
add_test(NAME cli_check_config_missing_model COMMAND lisa --check-config --model /nonexistent.gguf)
set_tests_properties(cli_check_config_missing_model PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_refuses_weak_key_off_loopback
         COMMAND lisa --check-config --model ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp --host 0.0.0.0)
set_tests_properties(cli_refuses_weak_key_off_loopback PROPERTIES
    ENVIRONMENT "LISA_API_KEY=short" WILL_FAIL TRUE)
add_test(NAME cli_rejects_invalid_env_value COMMAND lisa --check-config --model ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
set_tests_properties(cli_rejects_invalid_env_value PROPERTIES
    ENVIRONMENT "LISA_N_THREADS=lots" WILL_FAIL TRUE)
