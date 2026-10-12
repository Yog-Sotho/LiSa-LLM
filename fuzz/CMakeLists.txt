# libFuzzer targets (clang only): cmake --preset fuzz && cmake --build --preset fuzz
# Run: build/fuzz/fuzz/fuzz_request -dict=fuzz/request.dict fuzz/corpus/request
foreach(target fuzz_request fuzz_worker_reply)
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target} PRIVATE lisa_core)
    lisa_configure_target(${target})
    target_link_options(${target} PRIVATE -fsanitize=fuzzer)
endforeach()
